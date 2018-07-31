#include "ArrogantDetector.hpp"
#include "ShapeFunction.hpp"
#include "PowerSpectrum.hpp"
#include "kdp/kdpVectors.hpp"
#include "kdp/kdpHistogram.hpp"
#include <fstream>
#include <Qt/QtCore>

/* This file tests the stability and accuracy of the various shape functions.
 * Especially important is that the shapes can accept parameters which 
 * map to isotropic or delta-distribution shapes (or very close to these). 
 * 
 * Rigorously tested (25.07.2018 @ 14:41); stability vastly improved.
 * Values match Mathematica (hBoost.nb). Good enough for now.
*/ 

int main()
{
	QSettings settings("tester_ArrogantDetector.conf", QSettings::NativeFormat);
	std::string const allTower_filePath = settings.value("allTowers", "allTowers.dat").toString().toStdString();
	
	ArrogantDetector* detector = ArrogantDetector::NewDetector(settings);
	
	printf("etaMax - cal: % .3e  trk: % .3e\n", 
		detector->GetSettings().etaMax_cal, detector->GetSettings().etaMax_track);
	
	std::ofstream allTowers_file(allTower_filePath, std::ios::trunc);
	
	auto allTowers = detector->GetAllTowers();
	
	char buffer[1024];
	
	std::set<double> fracA;
	
	for(auto const& tower : allTowers)
	{
		sprintf(buffer, "% .3e % .3e % .3e % .3e\n", 
			tower.eta_lower, tower.eta_upper, tower.phi_lower, tower.phi_upper);
		allTowers_file << buffer;
		fracA.insert(tower.energy);
	}
	
	kdp::BinSpecs fBins("f", 20, {*fracA.begin(), *fracA.rbegin()});
	kdp::Histogram1 fA("fractionalArea", fBins);
	
	for(auto const& tower : allTowers)
		fA.Fill(tower.energy);	
	
	delete detector;
	return 0;
}
