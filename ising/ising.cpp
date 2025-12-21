#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <sstream>
namespace fs = std::filesystem;

#define UP 0
#define RIGHT 1
#define LEFT 2
#define DOWN 3

#ifndef L
#define L 100
#endif
#define SIZE L*L

#define DATA 10

#define MAG 0
#define MAG2 1
#define MAG4 2
#define MAGERR 3
#define SUSERR 4
#define ENE 5
#define ENE2 6
#define ENE4 7
#define ENERR 8
#define CHERR 9

using namespace std;

// Monte Carlo cadence (expressed in sweeps where 1 sweep = SIZE single-spin flips).
// For L=100, keep these modest unless you explicitly need uncorrelated samples.
const int THERMALIZATION_SWEEPS = 50;
const int SWEEPS_BETWEEN_SAMPLES = 5;
const int THERMALIZATION_FLIPS = THERMALIZATION_SWEEPS * SIZE;
const int FLIPS_BETWEEN_SAMPLES = SWEEPS_BETWEEN_SAMPLES * SIZE;

// Wolff cadence (cluster updates). One cluster update is not one sweep; tune as needed.
const int WOLFF_THERMALIZATION_STEPS = 50;
const int WOLFF_STEPS_BETWEEN_SAMPLES = 10;

// When writing configurations, map spins to {-1,+1} and optionally align the Z2 symmetry
// so that the written configuration has non-negative magnetization.
const bool ALIGN_Z2_ON_WRITE = true;
static fs::path g_out_dir;

static fs::path resolve_output_dir()
{
    const fs::path cwd = fs::current_path();
    const fs::path ising_dir = cwd / "Ising";
    if (fs::exists(ising_dir) && fs::is_directory(ising_dir)) return ising_dir;
    return cwd;
}



void initialize(bool spins[SIZE], mt19937& gen, uniform_int_distribution<int>& brandom);
void get_neighbors(int neighs[SIZE][4]);


void do_step(bool spins[SIZE],  int neighs[SIZE][4], double tstar, int N, double h[5], double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATA], ofstream* timeseries, ofstream* configs);
void do_step_wolff(bool spins[SIZE],  int neighs[SIZE][4], double tstar, int N, double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATA], ofstream* timeseries, ofstream* configs);
double magnetization(bool spins[SIZE]);

void flip_spin(bool spins[SIZE], int neighs[SIZE][4], double h[5], double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos);
void add_to_cluster(bool spins[SIZE], int neighs[SIZE][4], int pos, double& energy, double p, mt19937& gen, uniform_real_distribution<double>& ran_u);
double get_energy(bool spins[SIZE], int neighs[SIZE][4]);

void write(bool spins[SIZE]);

void w_output(ofstream& file, double tstar, int N, double m[DATA]);

struct IsingConfigHeader
{
    char magic[8];
    int32_t lattice_L;
    int32_t n_temps;
    int32_t n_samples;
    int32_t align_z2;
};

static inline double round6(double x)
{
    return std::round(x * 1e6) / 1e6;
}

static vector<double> build_temperature_grid()
{
    // Dense grid around the Ising transition; adjust as needed.
    const double tmin = 1.5;
    const double tmax = 3.2;
    const double dt = 0.01;

    vector<double> temps;
    for (double t = tmin; t <= tmax + 1e-12; t += dt)
    {
        temps.push_back(round6(t));
    }

    sort(temps.begin(), temps.end());
    temps.erase(unique(temps.begin(), temps.end()), temps.end());
    return temps;
}

static void write_temperature_header(ofstream& configs, double tstar)
{
    float tf = static_cast<float>(tstar);
    configs.write(reinterpret_cast<const char*>(&tf), sizeof(tf));
}

static std::string out_name(const std::string& stem, const std::string& ext)
{
    std::ostringstream oss;
    oss << stem << "_L" << L << ext;
    return oss.str();
}


int main(void)
{
    bool spins[SIZE]; //Stores the spins

	int i,j; //Counters

	int N; //Number of averages done into the system

	double h[5]; //Values of the exp(-J/kT)

	double energy; //Value of the energy of the system

    double m[DATA]; //This array contains several moments of the magnetization and energy

	int neighs[SIZE][4]; //To store nearest neighbours

    mt19937 gen(958431198); //Mersenne Twister RNG
	uniform_int_distribution<int> brandom(0, 1); //Get any random integer
	uniform_int_distribution<int> ran_pos(0, SIZE-1); //Get any random integer
	uniform_real_distribution<double> ran_u(0.0, 1.0); //Our uniform variable generator

	double tstar; //Control parameter
	double tcrit_up, tcrit_down; //Interval where we apply Wolff

    // Use Wolff only in a narrow band around Tc; Metropolis elsewhere.
    tcrit_up = 2.40;
    tcrit_down = 2.20;

	ofstream output; //Output of the stream
    ofstream timeseries_metro;
    ofstream timeseries_wolff;
    ofstream configs;

    initialize(spins, gen, brandom); //Init randomly
	get_neighbors(neighs); //Get neighbour table
	energy = get_energy(spins, neighs); //Compute initial energy

    g_out_dir = resolve_output_dir();

    // Samples per temperature
    N = 100;

    // Time series (per-measurement) output. This will be large: ~N * number_of_temperatures lines.
    timeseries_metro.open((g_out_dir / out_name("timeseries_metropolis", ".txt")).string());
    timeseries_metro << "# tstar sample_index sweeps_elapsed magnetization energy" << endl;
    timeseries_wolff.open((g_out_dir / out_name("timeseries_wolff", ".txt")).string());
    timeseries_wolff << "# tstar sample_index cluster_updates magnetization energy" << endl;

    // Spin configuration output (binary). Layout:
    // [header] then for each temperature:
    //   float32 T
    //   N records of dtype [('spins', int8, L*L), ('mabs', float32), ('energy', float32)]
    vector<double> temps = build_temperature_grid();
    configs.open((g_out_dir / out_name("ising_configs", ".bin")).string(), ios::binary);
    IsingConfigHeader header = {
        {'I', 'S', 'C', 'F', 'G', '1', '\0', '\0'},
        static_cast<int32_t>(L),
        static_cast<int32_t>(temps.size()),
        static_cast<int32_t>(N),
        static_cast<int32_t>(ALIGN_Z2_ON_WRITE ? 1 : 0),
    };
    configs.write(reinterpret_cast<const char*>(&header), sizeof(header));

	output.open((g_out_dir / out_name("test_alt162", ".txt")).string());
    for (auto it = temps.rbegin(); it != temps.rend(); ++it)
    {
        tstar = *it;
        write_temperature_header(configs, tstar);
        if (tstar <= tcrit_up && tstar >= tcrit_down)
        {
            do_step_wolff(spins, neighs, tstar, N, energy, gen, ran_u, ran_pos, m, &timeseries_wolff, &configs);
        }
        else
        {
            do_step(spins, neighs, tstar, N, h, energy, gen, ran_u, ran_pos, m, &timeseries_metro, &configs);
        }
        w_output(output, tstar, N, m);
        cout << tstar << endl;
    }
	output.close();
    timeseries_metro.close();
    timeseries_wolff.close();
    configs.close();

	return 0;

}

void write(bool spins[SIZE])
{
    int i;
    ofstream output;
    fs::path out = g_out_dir.empty() ? fs::path(".") : g_out_dir;
    output.open((out / "check.txt").string());
    for (i=0; i < SIZE; i++)
    {
        if (i % SIZE == 0 and i != 0) output << endl;
        output << spins[i] << " ";
    }
    return;
}

//Initialices the grid in which we are going to do the Ising, using random values
void initialize(bool spins[SIZE], mt19937& gen, uniform_int_distribution<int>& brandom)
{
	int i,j;

	//Init spins with a random distribution
	for (i=0; i < SIZE; i++)
	{
        spins[i] = brandom(gen); //Generate numbers
	}

	return;
}

//Fills the neigbour table
void get_neighbors(int neighs[SIZE][4])
{
	int i,j;
	int u,d,r,l;

	for (i=0; i < L; i++)
	{
		for (j=0; j < L; j++)
		{
		    //Get the (x,y) with periodic boundaries
			u = j+1 == L ? 0 : j+1;
			d = j-1 == -1 ? L-1 : j-1;
			r = i+1 == L ? 0 : i+1;
			l = i-1 == -1 ? L-1 : i-1;

            //(x,y) to index notation and store in table
			neighs[i+j*L][UP] = i+u*L;
			neighs[i+j*L][DOWN] = i+d*L;
			neighs[i+j*L][RIGHT] = r+j*L;
			neighs[i+j*L][LEFT] = l+j*L;
		}
	}

	return;
}

//Do all the things needed for a certain temperature
void do_step(bool spins[SIZE],  int neighs[SIZE][4], double tstar, int N, double h[5], double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATA], ofstream* timeseries, ofstream* configs)
{

	int i,j; //Counters
	double sum;//To compute the sum of spins
    double energysum;
    double chi, heat;
	double old_sum, old_chi, old_heat, old_energy;

    for (i=0; i < DATA; i++) m[i] = 0.0; //Init the values

	//Compute the factors of exp(-dH/kT)
	for (i=-4; i <= 4; i += 2)
	{
		h[(i+4)/2] =  min(1.0, exp(- 2.0 * i / tstar));
	}

	//Thermalize the state
	for (j=0; j < THERMALIZATION_FLIPS; j++)
	{
		flip_spin(spins, neighs, h,  energy, gen, ran_u, ran_pos);
	}

	///----- TODO: optimize the number of steps for thermalization/measures
	old_sum = 0.0;
	old_chi = 0.0;
	old_heat = 0.0;
	old_energy = 0.0;
	for (i=0; i < N; i++)
	{
		//Make changes and then average
		for (j=0; j < FLIPS_BETWEEN_SAMPLES; j++)
		{
			flip_spin(spins, neighs, h,  energy, gen, ran_u, ran_pos);
		}

        //Compute quantities at time j
        double mag = magnetization(spins);
        sum = abs(mag);
        chi = sum * sum;
        heat = energy * energy;

        //Add all the quantities
		m[MAG] += sum; //Magnetization
		m[MAG2] += chi; //For the susceptibility
		m[MAG4] += chi * chi; //For the Binder cumulant and also variance of susceptibility
		m[ENE] += energy; //Energy
		m[ENE2] += heat; //For specific heat
		m[ENE4] += heat * heat; //For the variance of specific heat
		//This are used for errors,
		m[MAGERR] += old_sum * sum; //in magnetization
		m[SUSERR] += old_chi * chi; //in susceptibility
		m[ENERR] += old_energy * energy; //in energy
		m[CHERR] += old_heat * heat; //in specific heat

		//Get the value for the next iteration
		old_sum = sum;
		old_energy = energy;
		old_chi = chi;
		old_heat = heat;

        // Optionally store the raw time series for autocorrelation analysis.
        if (timeseries && timeseries->is_open())
        {
            double sweeps_elapsed = (THERMALIZATION_FLIPS + (i+1) * FLIPS_BETWEEN_SAMPLES) / (1.0 * SIZE);
            (*timeseries) << fixed << setprecision(6)
                          << tstar << " " << i << " " << sweeps_elapsed << " " << sum << " " << energy << endl;
        }

        // Optionally store the raw configuration (int8 spins in {-1,+1}) plus (mabs, energy).
        if (configs && configs->is_open())
        {
            static vector<int8_t> buf;
            if (buf.size() != SIZE) buf.assign(SIZE, 0);
            const bool flip = ALIGN_Z2_ON_WRITE && (mag < 0.0);
            for (int s = 0; s < SIZE; s++)
            {
                const int8_t val = spins[s] ? 1 : -1;
                buf[s] = flip ? static_cast<int8_t>(-val) : val;
            }
            configs->write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            float mabs_f = static_cast<float>(sum);
            float e_f = static_cast<float>(energy);
            configs->write(reinterpret_cast<const char*>(&mabs_f), sizeof(mabs_f));
            configs->write(reinterpret_cast<const char*>(&e_f), sizeof(e_f));
        }
	}

    //Finish the average
    for (i=0; i < DATA; i++)  m[i] /= (1.0 * N);

	return;
}


//Do all the things needed for a certain temperature
void do_step_wolff(bool spins[SIZE],  int neighs[SIZE][4], double tstar, int N, double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATA], ofstream* timeseries, ofstream* configs)
{

	int i,j; //Counters
	double sum; //To compute the sum of spins
    double chi, heat; //To compute magnetic susceptibility
	//Note: we use directly variable energy for the energy
    //To remember last results
	double old_sum, old_chi, old_heat, old_energy;


	double pa = 1.0 - exp(- 2.0 / tstar); //TODO change

    for (i=0; i < DATA; i++) m[i] = 0.0; //Init the values

    //Make changes and then average
    for (j=0; j < WOLFF_THERMALIZATION_STEPS; j++) add_to_cluster(spins, neighs, ran_pos(gen), energy, pa, gen, ran_u);

	///----- TODO: optimize the number of steps for thermalization/measures
	old_sum = 0.0;
	old_chi = 0.0;
	old_heat = 0.0;
	old_energy = 0.0;


	for (i=0; i < N; i++)
	{
		//Make changes and then average
        for (j=0; j < WOLFF_STEPS_BETWEEN_SAMPLES; j++) add_to_cluster(spins, neighs, ran_pos(gen), energy, pa, gen, ran_u);

	        //Compute quantities at time j
            double mag = magnetization(spins);
	        sum = abs(mag);
	        chi = sum * sum;
	        heat = energy * energy;

        //Add all the quantities
		m[MAG] += sum; //Magnetization
		m[MAG2] += chi; //For the susceptibility
		m[MAG4] += chi * chi; //For the Binder cumulant and also variance of susceptibility
		m[ENE] += energy; //Energy
		m[ENE2] += heat; //For specific heat
		m[ENE4] += heat * heat; //For the variance of specific heat
		//This are used for errors,
		m[MAGERR] += old_sum * sum; //in magnetization
        m[ENERR] += old_energy * energy; //in energy
		m[SUSERR] += old_chi * chi; //in susceptibility
		m[CHERR] += old_heat * heat; //in specific heat

		//Get the value for the next iteration
		old_sum = sum;
        old_energy = energy;
		old_chi = chi;
		old_heat = heat;

	        if (timeseries && timeseries->is_open())
	        {
	            int cluster_updates = WOLFF_THERMALIZATION_STEPS + (i+1) * WOLFF_STEPS_BETWEEN_SAMPLES;
	            (*timeseries) << fixed << setprecision(6)
	                          << tstar << " " << i << " " << cluster_updates << " " << sum << " " << energy << endl;
	        }

            if (configs && configs->is_open())
            {
                static vector<int8_t> buf;
                if (buf.size() != SIZE) buf.assign(SIZE, 0);
                const bool flip = ALIGN_Z2_ON_WRITE && (mag < 0.0);
                for (int s = 0; s < SIZE; s++)
                {
                    const int8_t val = spins[s] ? 1 : -1;
                    buf[s] = flip ? static_cast<int8_t>(-val) : val;
                }
                configs->write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
                float mabs_f = static_cast<float>(sum);
                float e_f = static_cast<float>(energy);
                configs->write(reinterpret_cast<const char*>(&mabs_f), sizeof(mabs_f));
                configs->write(reinterpret_cast<const char*>(&e_f), sizeof(e_f));
            }
		}

    //Finish the average
    for (i=0; i < DATA; i++)  m[i] /= (1.0 * N);

	return;
}

//Flip a spin via Metropolis
void flip_spin(bool spins[SIZE], int neighs[SIZE][4], double h[5], double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos)
{
    int index = ran_pos(gen); //Get a random position to flip
    //Compute the sum of neighbours
    int sum_neigh = spins[neighs[index][UP]] + spins[neighs[index][DOWN]] + spins[neighs[index][RIGHT]] + spins[neighs[index][LEFT]];
    //Use this to get the energy change (depending on the value of my spin)
    int change = spins[index] ? 2.0 * (sum_neigh) - 4.0 : 4.0 - 2.0 * (sum_neigh);

    //Apply Metropolis skim
    if (ran_u(gen) < h[(change+4)/2])
    {
        spins[index] = !spins[index];
        energy += (2.0*change)/(1.0*SIZE);
        //cout << change << "  " << (2.0*change)/(1.0*SIZE) << "  " << energy << endl;
    }

    return;
}

//Compute the magnetization
double magnetization(bool spins[SIZE])
{
    int i;

    double sum = 0.0;
    //Sum all the values of the spins
    for (i=0; i < SIZE; i++)
    {
        sum += spins[i];
    }
    //And then return them
    return (2.0*sum - SIZE)/(SIZE);
}

//Computes the energy of the system
double get_energy(bool spins[SIZE], int neighs[SIZE][4])
{
    int i; //Counters
    int sum_neigh;

    int energy = 0; //Sum

    //For every spin,
    for (i=0; i < SIZE; i++)
    {
        //Get sum of the neighbours
        sum_neigh = spins[neighs[i][UP]] + spins[neighs[i][DOWN]] + spins[neighs[i][RIGHT]] + spins[neighs[i][LEFT]];
        //And compute the energy change
        energy += spins[i] ? 2.0 * (sum_neigh) - 4.0 : 4.0 - 2.0 * (sum_neigh);
    }

    return 2.0*energy/(1.0*SIZE); //Return the energy
}

// Executes Wolff algorithm (iterative to avoid deep recursion for large L).
void add_to_cluster(bool spins[SIZE], int neighs[SIZE][4], int pos, double& energy, double p, mt19937& gen, uniform_real_distribution<double>& ran_u)
{
    static vector<int> stack;
    static vector<int32_t> mark;
    static int32_t stamp = 1;

    if (mark.size() != SIZE)
    {
        mark.assign(SIZE, 0);
        stack.reserve(SIZE);
    }

    stamp++;
    if (stamp == numeric_limits<int32_t>::max())
    {
        fill(mark.begin(), mark.end(), 0);
        stamp = 1;
    }

    const bool seed_spin = spins[pos];

    stack.clear();
    stack.push_back(pos);
    mark[pos] = stamp;

    while (!stack.empty())
    {
        int cur = stack.back();
        stack.pop_back();

        int sum_neigh = spins[neighs[cur][UP]] + spins[neighs[cur][DOWN]] + spins[neighs[cur][RIGHT]] + spins[neighs[cur][LEFT]];
        int delta_energy = spins[cur] ? 2.0 * (sum_neigh) - 4.0 : 4.0 - 2.0 * (sum_neigh);

        energy += (2.0 * delta_energy) / (1.0 * SIZE);
        spins[cur] = !spins[cur];

        for (int i = 0; i < 4; i++)
        {
            int n = neighs[cur][i];
            if (mark[n] == stamp) continue;
            if (spins[n] != seed_spin) continue;
            if (ran_u(gen) >= p) continue;
            mark[n] = stamp;
            stack.push_back(n);
        }
    }
}

void w_output(ofstream& file, double tstar, int N, double m[DATA])
{
    file << tstar << " " << 1.0/tstar << " "; //Write T and B

    //We here take in account residual errors, which, for low T, makes the quantities chi, ch, etc.
    //to diverge. This must be substracted. That's why we use an abs for correlation time and also
    //a check to avoid zero value of variances.

    //Then write the quantities and the corresponding errors to a file. The four things are equal,
    //but each one referred to a different quantity.

    double chi = m[MAG2] - m[MAG] * m[MAG]; //Magnetic susceptibility (up to T factor)
    double rhom = chi != 0 ? (m[MAGERR] - m[MAG] * m[MAG]) / chi : 0.0; //Rho magnetization, computed if chi != 0
    double taugm = rhom != 1.0 ? rhom / (1.0 - rhom) : 0.0; //Taug magnetization, computed if rhom != 0
    file << m[MAG] << " " << sqrt(chi * abs(2.0 * taugm + 1) / (1.0*N)) << " "; //Write everything

    double fourth = m[MAG4] - m[MAG2] * m[MAG2]; //Susceptibility variance
    double rhos = fourth != 0.0 ? (m[SUSERR] - m[MAG2] * m[MAG2]) / fourth : 0.0; //Rho susceptibility
    double taugs = rhos != 1.0 ? rhos /(1.0 - rhos) : 0.0; //Taug susceptibility
    double error_sq = sqrt(fourth * abs(2.0 * taugs + 1) / (1.0*N));
    file << " " << chi << " " << error_sq << " ";

    double heat = m[ENE2] - m[ENE] * m[ENE]; //Specific heat (up to T^2 factor)
    double rhoe = heat != 0.0 ? (m[ENERR] - m[ENE]*m[ENE]) / heat : 0.0;
    double tauge = rhoe != 1.0 ? rhoe / (1.0 - rhoe) : 0.0;
    file << " " << m[ENE] << " " << sqrt(heat * abs(2.0 * tauge + 1) / (1.0*N)) << " ";

    double fourth_ene = m[ENE4] - m[ENE2] * m[ENE2];
    double rhoc = fourth_ene != 0.0 ? (m[CHERR] - m[ENE2] * m[ENE2]) / fourth_ene : 0.0;
    double taugc = rhoc != 1.0 ? rhoc / (1.0 - rhoc) : 0.0;
    file << " " << heat << " " << sqrt(fourth_ene * abs(2.0 * taugc + 1) / (1.0*N)) << " ";

    //Binder cumulant
    double binder = 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2]); //Computes 4th cumulant minus one, b-1.
    file << binder << " " << 2.0 * (1.0 - binder) * (error_sq / m[MAG2]) << endl;
    return;
}
