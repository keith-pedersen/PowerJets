#include "LHE_Pythia_PowerJets.hpp"

int main()
{
	LHE_Pythia_PowerJets test;
	for(size_t i = 0; i < 100; ++i)
		test.Next();
		
	auto const& H_l = test.Get_H_det();
	
	for(size_t lMinus1 = 0; lMinus1 < H_l.size(); ++lMinus1)
		printf("%lu %.3e\n", lMinus1 + 1, H_l[lMinus1]);
}
