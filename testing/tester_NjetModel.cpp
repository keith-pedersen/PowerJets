#include "NjetModel.hpp"

int main()
{
	ShowerParticle root({0.5, 0.9, 0.3, 0.9, 1.},{{false}}, {0.2, 0.5, 1.});
	
	auto jets = root.GetJets();
	
	for(auto const& jet : jets)
		printf("%.3e %.3e %.3e %.3e\n", jet.p4.x0, jet.p4.x1, jet.p4.x2, jet.p4.x3);
	
	return 0;
}
