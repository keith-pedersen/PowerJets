#include "LHE_Pythia_PowerJets.hpp"
#include "pqRand/pqRand.hpp"

int main()
{
	size_t const lMax = 36;
	size_t const nExtraJets = 0;
	size_t const nParams = 2 + 3 + 4*nExtraJets;
	std::vector<std::vector<bool>> addresses = {{true},};
	
	pqRand::engine gen;
	
	LHE_Pythia_PowerJets test;
	for(size_t i = 0; i < 1000; ++i)
	{
		test.Next();
		
		double Hl_Obs_error = 1.;
		double Hl_Hybrid_error = 1.;
		
		auto Hl = test.Get_Hl_Obs(lMax);
				
		Hl_Obs_error = std::sqrt(std::accumulate(Hl.begin(), Hl.end(), 0.,
			[](double const sum, double const val){return sum + kdp::Squared(val);}));		
		
		std::vector<double> params;
		
		for(size_t k = 0; k < nParams; ++k)
		{
			if(k == 0)
				params.push_back(0.3 + 0.7 * gen.U_even());
			else
				params.push_back(gen.U_even());
		}
					
		auto jets = ShowerParticle(params, addresses).GetJets();
		
		Hl = test.Get_Hl_Hybrid(lMax, jets);
				
		Hl_Hybrid_error = std::sqrt(std::accumulate(Hl.begin(), Hl.end(), 0.,
			[](double const sum, double const val){return sum + kdp::Squared(val);}));
			
		Hl = test.Get_Hl_Jet(lMax, jets);
				
		double const Hl_Jet_error = std::sqrt(std::accumulate(Hl.begin(), Hl.end(), 0.,
			[](double const sum, double const val){return sum + kdp::Squared(val);}));
		
		//~ if((Hl_Obs_error > 1e-8) or (Hl_Hybrid_error > 1e-8) or (Hl_Jet_error > 1e-8))
		printf("%lu  %.3e  %.3e  %.3e\n", i, Hl_Obs_error, Hl_Hybrid_error, Hl_Jet_error);
	}
		
	//~ std::vector<double> params = {0.3, 0.8, 0.3, 0.9, 0.9};
	//~ std::vector<std::vector<bool>> addresses = {{false},};
	//~ auto const jets = ShowerParticle(params, addresses).GetJets();
	
	//~ for(auto const& jet : jets)
		//~ printf("%.5e\n", jet.p4.p().Mag() / jet.p4.x0);
		
	//~ auto const& H_l = test.Get_Hl_Hybrid(64, jets); //test.Get_Hl_Obs(64);
	//~ std::cout << "Done orig\n";
	//~ auto const H_l_mk2 = test.Get_Hl_Hybrid(64, jets); // test.Get_Hl_Obs_mk2(64);
	//~ std::cout << "Done parallel\n";
		
	//~ for(size_t lMinus1 = 0; lMinus1 < H_l.size(); ++lMinus1)
		//~ printf("%lu %.3e %.3e\n", lMinus1 + 1, H_l[lMinus1], H_l_mk2[lMinus1]);
}
