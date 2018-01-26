#include "NjetModel.hpp"
#include "pqRand/pqRand.hpp"
//~ #include "kdp/kdpVectors.hpp"

void PrintVec4(kdp::Vec4 const& p4)
{
	printf("[%.16e (%.16e, %.16e, %.16e)]\n", p4.x0, p4.x1, p4.x2, p4.x3);
}

int main()
{
	std::vector<std::vector<bool>> addresses = {{true}, {false}, {false, true}, {false, true, true}};

	NjetModel modeler;
	pqRand::engine gen;
	
	for(size_t i = 0; i < 100; ++i)
	{
		std::vector<double> params;
		
		for(size_t k = 0; k < 17; ++k)
			params.push_back(gen.U_even());
			
		for(size_t j = 0; j < 3; ++j)
		{
			auto H_l = modeler.H_l(ShowerParticle(params, addresses).GetJets(), 128, 1e5);
			printf("%.3e\n", H_l.back());
				
			if(std::isnan(H_l.back()))
			{
				for(size_t k = 0; k < 17; ++k)
					printf("%.16e, ", params[k]);
				printf("\n");
			}
		}
	}		
}
