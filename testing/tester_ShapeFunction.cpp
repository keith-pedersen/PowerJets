#include "ShapeFunction.hpp"
#include "PowerSpectrum.hpp"
#include "kdp/kdpVectors.hpp"
#include <fstream>

/* This file tests the stability and accuracy of the various shape functions.
 * Especially important is that the shapes can accept parameters which 
 * map to isotropic or delta-distribution shapes (or very close to these). 
 * 
 * Rigorously tested (25.07.2018 @ 14:41); stability vastly improved.
 * Values match Mathematica (hBoost.nb). Good enough for now.
*/ 

int main()
{
	size_t const lMax = 20000;
	
	// First test extreme shapes
	{
		h_Cap iso(1.);
		h_Cap almostIso(1.-1e-6);
		h_Cap almostDelta(1e-6);
		h_Cap delta(0.);
		
		PowerSpectrum::WriteToFile("h_Cap_test.dat", 
			{iso.hl_Vec(lMax), almostIso.hl_Vec(lMax), almostDelta.hl_Vec(lMax), delta.hl_Vec(lMax)}, 
			"iso almostIso almostDelta delta");
	}
	
	{
		h_Gaussian iso(1e300);
		h_Gaussian almostIso(1e3);
		h_Gaussian almostDelta(1e-3);
		h_Gaussian delta(0.);
	
		PowerSpectrum::WriteToFile("h_Gaussian_test.dat",
			{iso.hl_Vec(lMax), almostIso.hl_Vec(lMax), almostDelta.hl_Vec(lMax), delta.hl_Vec(lMax)}, 
			"iso almostIso almostDelta delta");
	}
		
	{
		kdp::Vec3 p3(0., 0., 1.);
		
		// Discovered that boost is not very stable with the existing stability criteria
		h_Boost iso(kdp::Vec3(0., 0., 0.), 1.);
		h_Boost almostIso(p3, 1e2);
		h_Boost almostDelta(p3, 1e-2);
		h_Boost delta(p3, 0.);
	
		PowerSpectrum::WriteToFile("h_Boost_test.dat",
			{iso.hl_Vec(lMax), almostIso.hl_Vec(lMax), almostDelta.hl_Vec(lMax), delta.hl_Vec(lMax)}, 
			"iso almostIso almostDelta delta");
	}
	
	// Now test normal shapes (confirmed (25.07.2018 @ 14:27)

	auto cap = h_Cap(1e-3); // Cap confirmed
	auto boost = h_Boost(kdp::Vec3(0.9, 0.9, 0.9), 1.5); // Boost confirmed
	auto gauss = h_Gaussian(1e-1); // Gauss confirmed
	
	PowerSpectrum::WriteToFile("hl_test.dat", 
			{cap.hl_Vec(lMax), boost.hl_Vec(lMax), gauss.hl_Vec(lMax)}, 
			"cap(surfFrac=1e-3) almostIso(p~=m) h_Gauss(lambda = 1e-1)");
	
	return 0;
}
