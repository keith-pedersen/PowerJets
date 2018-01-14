#include "NjetModel.hpp"

int main(int argCount, char** argVec)
{
	using vec4_t = NjetModel::vec4_t;
	
	double const granularity = (argCount > 1) ? atof(argVec[1]) : 1e3;
	//~ printf("granularity: %.3e\n", granularity);
	
	pqRand::engine gen;
	
	//~ auto isoVec = NjetModel::IsoCM(4, gen);
	
	NjetModel modeler;
	std::vector<Jet> jetVec;
	
	//~ for(auto const& p3 : isoVec)
		//~ jetVec.emplace_back(p3, 2e-1);
		
	jetVec.emplace_back(1., 2., 3., 0.1);
	jetVec.emplace_back(-1., 2.1, -3., 0.1);
	jetVec.emplace_back(0., -4., 0., 0.1);
		
	auto H_l = modeler(jetVec, 1024, granularity);
			
	printf("%lu  %.3e\n", 0, 1.);
	for(size_t lMinus1 = 0; lMinus1 < H_l.size(); ++lMinus1)
		printf("%lu  %.3e\n", lMinus1 + 1, H_l[lMinus1]);
	
	return 0;
}
