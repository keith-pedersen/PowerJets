#include "ArrogantDetector.hpp"
#include "ShapeFunction.hpp"
#include "PowerSpectrum.hpp"
#include "kdp/kdpVectors.hpp"
#include "kdp/kdpHistogram.hpp"
#include <fstream>
#include <Qt/QtCore>

/*! @file tester_ArrogantDetector.cpp
 *  @brief This file tests the calorimeter grid defined by ArrogantDetector, 
 *  outputting the generated towers to a file which can be plotted with Mathematica
*/

int main()
{
	QSettings settings("tester_ArrogantDetector.ini", QSettings::IniFormat);
	std::string const allTower_filePath = settings.value("allTowers", "allTowers.dat").toString().toStdString();
	
	ArrogantDetector* detector = ArrogantDetector::NewDetector(settings);
	
	printf("etaMax - cal: % .3e  trk: % .3e\n", 
		detector->GetSettings().etaMax_cal.value, detector->GetSettings().etaMax_track.value);
	
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
