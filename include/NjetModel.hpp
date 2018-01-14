#ifndef N_JET_MODEL
#define N_JET_MODEL

#include "SpectralPower.hpp"
#include "RecursiveLegendre.hpp"
#include "pqRand/pqRand.hpp"
//~ #include "ArrogantDetector.hpp"
#include <vector>

/*! @brief A 4-momenta with a shape (defined in its CM frame and boosted into the lab frame).
 * 
 *  At the moment the jet's shape is always isotropic,
 *  but the interface allows more sophisticated (though azimuthally symmetric) shapes 
 *  without altering the API.
*/
class Jet
{
	public: 
		using real_t = SpectralPower::real_t;
		using vec3_t = kdp::Vector3<real_t>; //!< @brief The 3-vector type
		using vec4_t = kdp::Vector4<real_t>; //!< @brief The 4-vector type
		//! @brief Random samples from the jet shape are generated in increments of this size
		static constexpr size_t incrementSize = size_t(1) << 8; // 512, not too large, not too small
		using incrementArray_t = std::array<real_t, incrementSize>;
		
		vec4_t p4; //!> @brief The 4-momentum
		real_t mass; //!> @brief The mass is stored separately to avoid re-computation (with rounding error).
						
		/*! @brief The azimuthally symmetric jet shape function.
		 *
		 *  A normalized PDF for \p z_CM = cos(theta_CM), the CM-frame polar angle
		 *  (with the +z direction defined by jet's lab-frame 3-momentum).
		*/
		real_t h_CM(real_t const z_CM) const {return real_t(0.5);}
		
		/*! @brief Randomly sample the jet's shape in the lab frame, via unit vectors.
		 * 
		 *  The jet is oriented in the lab frame so that it is parallel to +z.
		 *  Each sample (x, y, z) is returned via its y and z
		 *  (being unit vectors, their x position is calculable from the other two).
		 *  
		 *  The variates are sampled using antithetic variance reduction;
		 *  each uniform variate from \p gen is used to sample 
		 *  one forward (+z) and one backward (-z) variate in the CM frame, 
		 *  both of which are then boosted into the lab frame.
		*/ 
		void SampleShape(incrementArray_t& z_lab, incrementArray_t& y_lab, 
			pqRand::engine& gen) const;
		
		// This is the main ctor; all others call it.
		Jet(vec4_t const& p4_in, real_t const mass_in, std::vector<real_t> shapeParams = {}):
			p4(p4_in), mass(mass_in) {}
			
		Jet(vec3_t const& p3_in, real_t const mass_in, std::vector<real_t> shapeParams = {}):
			Jet(vec4_t(mass_in, p3_in, kdp::Vec4from2::Mass), mass_in, shapeParams) {}
				
		// The interface we expect to use from inside a Cython loop
		Jet(real_t const p1, real_t const p2, real_t const p3, real_t const mass_in, 
			std::vector<real_t> shapeParams = {}):
		Jet(vec3_t(p1, p2, p3), mass_in, shapeParams) {}
};

/*! @brief Calculate the spectral power H_l for an n-jet model.
 * 
 *  H_l is calculated from the coefficients rho_l^m when the 
 *  energy density rho is transformed into the basis of the spherical harmonics Y_l^m.
 *  If the energy density is built from N jets
 *  	rho = rho_(1) + rho_(2) + ... + rho_(N)
 *  Then the power spectrum H_l can be calculated by summing terms like
 *  	rho_(i)_l^m * rho_(j)_l^m   (Einstein summation over l and m)
 * 
 *  Each of these terms is rotationally invariant, and thus can be 
 *  calculated in whichever orientation simplifies the calculation the most.
 *  This happens to be the orientation where jet_(i) is rotated parallel to the +z axis, 
 *  with jet_(j) sticking off-axis at its theta_ij (and some arbitrary phi).
 *  If all jets are azimuthally symmetric (a critical assumption), 
 *  then only the m = 0 terms for jet_(i) are non-zero, 
 *  so the summation simplifies to
 * 	rho_(i)_l * rho_(j)_l   (the product of the m = 0 terms).
 * 
 *  The m = 0 density coefficient rho_(i)_l depends on the shape of the jet.
 *  An azimuthally symmetric jet's shape is defined by 
 *  its CM shape function h(z_CM) (a normalized PDF in z_CM = cos(theta_CM), see Jet::h_CM).
 *  This CM shape is then boosted into the lab frame.
 *  It turns out that rho_(i)_l are simply the coefficients for the 
 *  Legnedre expansion of the lab frame shape:
 *  	\rho_(i)_l = \int_{-1}^{1} h(z_lab) P_l(z_lab) d z_lab
 * 
 *  The Legendre coefficients are calculable for jet_(i) via a recursive formula.
 *  This formula has some stability issues, but is generally OK.
 *  However, the Legendre coefficeints for the off-z jet are non-trivial
 *  (due to the non-trivial geometry of a sphere). 
 *  We therefore calculate them using Monte Carlo integration, 
 *  since this scheme is quite conducive to the recursive definition of P_l.
 * 
 *  While it is possible to recursively compute \rho_(i)_l for jet shape h(z) = 0.5, 
 *  it is not trivial for more complicated h(z) (due to the mapping of the boost).
 *  Hence, because the Monte Carlo integration is totally general, 
 *  we use it even for the trivial jet shape.
*/
class NjetModel
{
	private:
		mutable pqRand::engine gen;
		
	public:
		using vec3_t = Jet::vec3_t;
		using vec4_t = Jet::vec4_t;
		using real_t = SpectralPower::real_t;
		//~ typedef SpectralPower::PhatF PhatF;
	
		NjetModel(QSettings const& settings);
		NjetModel(std::string const& iniFileName);
		NjetModel();
		~NjetModel();
		
		// jetParams is a list like {p1_x, p1_y, p1_z, m1, ...}
		std::vector<real_t> operator()(std::vector<Jet> const& jetVec_unsorted, 
			size_t const lMax, real_t const jetShapeGranularity) const;
					
		//~ static std::vector<vec4_t> GetJets(std::vector<real_t> const& jetParams);
		//~ static std::vector<std::vector<real_t>> GetJetsPy(std::vector<real_t> const& jetParams);
				
		//~ // Draw a unit vector isotropically from the unit sphere
		//~ static vec3_t IsoVec3(pqRand::engine& gen);

		//~ // Return n_request unit 3-vectors randomly but isotropically distributed, 
		//~ // but which nonetheless sum to zero. Due to the balancing scheme, 
		//~ // the number of vectors returned may occasionally be (n_request + 1)
		//~ static std::vector<kdp::Vec3> IsoCM(size_t const n_request, 
			//~ pqRand::engine& gen, real_t const tolerance = 1e-15);
};

#endif
