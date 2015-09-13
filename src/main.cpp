#include <iostream>

#include <boost/program_options.hpp>

void runCpu(boost::program_options::variables_map const& vm);
void runGpu(boost::program_options::variables_map const& vm);
void runDyn(boost::program_options::variables_map const& vm);

void parseParameters(int argc, char* argv[]) {
	namespace po = boost::program_options;

	po::options_description general("General Options");

	std::string type;

	general.add_options()
		("type", po::value<std::string>(&type), "\"cpu\", \"gpu\" or \"dyn\"")
		("width", po::value<int>()->default_value(4096), "width of the matrix")
		("height", po::value<int>()->default_value(4096), "height of the matrix")
		("numIterations", po::value<int>()->default_value(50), "number of iterations to calculate")
		("csv", po::value<std::string>()->default_value("output.csv"), "name of the csv file")
		("matrix", po::value<std::string>(), "name of the matrix output file")
	;

	po::options_description gpu("GPU Options");

	gpu.add_options()
		("kernel", po::value<int>()->default_value(1), "version of the kernel to use")
	;

	po::options_description all("Usage");

	all.add(general).add(gpu);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, all), vm);
	po::notify(vm);

	if (type == "cpu") {
		runCpu(vm);
	}
	else if (type == "gpu") {
		runGpu(vm);
	}
	else if (type == "dyn") {
		runDyn(vm);
	}
	else {
		std::cout << "Error: --type must be set to either cpu, gpu, or dyn.\n\n";
		std::cout << all << std::endl;
	}
}

int main(int argc, char* argv[]) {
	parseParameters(argc, argv);
}
