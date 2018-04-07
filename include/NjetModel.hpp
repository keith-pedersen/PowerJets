#ifndef N_JET_MODEL
#define N_JET_MODEL

#include "SpectralPower.hpp"
#include "RecursiveLegendre.hpp"
#include "pqRand/pqRand.hpp"
//~ #include "ArrogantDetector.hpp"
#include <vector>
#include <memory> // ShowerParticle
#include "fastjet/PseudoJet.hh"
#include "ShapeFunction.hpp"
#include <QtCore/QSettings>

//======================================================================

// Homo sapiens
//                      
//      ()      5 - head:   and the way.
//     /||\     4 - heart:  but you have the will
//    / || \    3 - hands:  It will need maintenance;
//     //\\     2 - heat:   you'll find your home.
//    //  \\    1 - feet:   After some searching,

//======================================================================

/*! @brief A 4-momenta with a cached mass.
 * 
 *  This is useful when the object is significantly boosted, 
 *  since calculating it's mass from E**2 - p**2 loses precision.
*/
struct Jet
{
	using real_t = PowerJets::real_t;
	using vec3_t = kdp::Vector3<real_t>; //!< @brief The 3-vector type
	using vec4_t = kdp::Vector4<real_t>; //!< @brief The 4-vector type
	
	vec4_t p4; //!> @brief The 4-momentum
	
	/*! @brief The mass is cached to avoid re-computation (with rounding error).
	 * 
	 *  \warning The mass is not actively managed after construction. 
	 *  If you operate on p4, you must recompute mass.
	*/
	real_t mass; 
	
	Jet():p4(), mass(0) {}
	
	explicit Jet(bool):p4(false) {}
	
	explicit Jet(vec4_t const& p4_in):
		p4(p4_in), mass(p4.Length()) {}
	
	// This is the main ctor, which all others should call.	
	explicit Jet(vec3_t const& p3_in, real_t const w0, kdp::Vec4from2 const w0type);
			
	// The interface we expect to use from inside a Cython loop
	explicit Jet(real_t const x1, real_t const x2, real_t const x3, 
		real_t const w0, kdp::Vec4from2 const w0type);
		
	Jet(fastjet::PseudoJet const& pj);
	
	//! @brief Rotate the z-axis to (sin(theta) cos(phi), sin(theta) sin(phi), cos(phi))
	template <class T>
	static void Rotate(std::vector<T>& jetVec, vec3_t const& axis, real_t const angle)
	{
		kdp::Rot3 rotator(axis, angle);
			
		for(auto& jet : jetVec)
			rotator(jet.p4.p());
	}
};

/*! @brief A 4-momenta with a shape (defined in its CM frame and boosted into the lab frame).
 * 
 *  At the moment the jet's shape is always isotropic,
 *  but the interface allows more sophisticated (though azimuthally symmetric) shapes 
 *  without altering the API.
*/
class ShapedJet : public Jet
{
	public:
		//! @brief The address of the jet in the splitting tree.
		std::vector<bool> address;
		h_Boost shape;
		
		using Jet::Rotate;		
	
		//! @brief Random samples from the jet shape are generated in increments of this size
		static constexpr size_t incrementSize = size_t(1) << 8; // 512, not too large, not too small
		using incrementArray_t = std::array<real_t, incrementSize>;
		
		// We assume that all shape parameters will be passed from a std::vector<real_t>
		using param_iter_t = std::vector<real_t>::const_iterator;
						
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
		
		ShapedJet(): Jet(), shape(real_t(0)) {} // A minimal nullary constructor so Cython can use push_back
				
		// The don't initialize constructor
		explicit ShapedJet(bool): Jet(false), shape(real_t(0)) {}
		
		ShapedJet(vec3_t const& p3_in, real_t const w0, kdp::Vec4from2 const w0type,
			std::vector<bool> address_in = std::vector<bool>(),
			std::vector<real_t> const& shapeParams = {}):
			Jet(p3_in, w0, w0type),
			address(std::move(address_in)),
			shape(vec4_t::BetaFrom_Mass_pSquared(mass, p4.p().Mag2())) {}
				
		// The interface we expect to use from inside a Cython loop
		ShapedJet(real_t const x1, real_t const x2, real_t const x3, 
			real_t const w0, kdp::Vec4from2 const w0type, 
			std::vector<bool> address_in = std::vector<bool>(),
			std::vector<real_t> const& shapeParams = {}):
		Jet(x1, x2, x3, w0, w0type),
		address(std::move(address_in)),
		shape(vec4_t::BetaFrom_Mass_pSquared(mass, p4.p().Mag2())) {}
		
		// We assume that the shape of the jet will be initialized later, 
		// after the jet's 4-vector and mass have been defined.
		void SetShape(param_iter_t const shapeParam_begin, param_iter_t const shapeParam_end);
		
		std::vector<real_t> OnAxis(size_t const lMax) const {return shape.OnAxis(lMax);}
		
		// cython does not support
		//~ static bool Sort_by_Mass(ShapedJet const& left, ShapedJet const& right)
		//~ {
			//~ return left.mass > right.mass;
		//~ }
		
		bool operator < (ShapedJet const& that) const;
};

// This class is a C++ implementation of particleJet_mk3.py

/*! @brief An extension of ShapedJet which manages a shower built from 
 *  mother particles splitting to two daughters (a -> bc).
 *  Each ShowerParticle is a node in a binary tree.
 * 
 *  The shower/splitting-tree will be determined by a list of parameters.
 *  When each mother is split, she has seom dimensionless energy f_a and mass q_a (both non-negative). 
 *  It's splitting is determined by 4 parameters (the 4 d.o.f for particle b, 
 *  with particle c defined by conservation of momentum).
 * 
 * 	z (the splitting fraction 0 <= z <= 1; f_b = z * f_a, f_c = (1-z) * f_a)
 * 	u_b (b's [non-negative] mass fraction; q_b = u_b * q)
 * 	u_c (c's [non-negative] mass fraction; q_c = u_c * q)
 * 	phi (the rotation of the daughter's splitting-plane relative to the mother's splitting-plane)
 * 	
 *  There are a number of constraints which are needed for each splitting.
 *  The daughter mass must conserve energy
 * 
 * 	u_b + u_c <= 1
 * 
 *  Also, z cannot arbitrarily satisfy p^mu conservation;
 *  it must satisfy z- <= z <= z+
 * 
 * 	z(+/-) = 1/2*[1 + (q_b**2 - q_a**2)] +/- beta_a * 
 * 		sqrt((1 + q_b + q_c)(1 + q_b - q_c)(1 - q_b + q_c)(1 - q_b - q_c))]
 * 
 *  We expect that the shower parameters will be determined from a 
 *  non-linear minimization fit. In general, while it is easy to 
 *  set "bounds" on parameters (i.e. param_i must exist in some domain), 
 *  it is non-trivial to enforce constraints (i.e. some equality of inequality 
 *  which depends on multiple parameters must be satisfied).
 *  To avoid constraints, we can build the constraints into the fitting parameters
 *  (so that, if the constraints are non-linear, the non-linearity is 
 *  baked into the fit parameters). We choose the following fit parameters.
 * 
 *         domain      parameter
 * 	==============================================================================================
 * 	[0, 1]        | uSum = u_b + u_c 
 * 	[0, 1]        | uFrac ==> u_b = uFrac * uSum
 * 	[0, 1]        | zStar ==> z = z(-) + zStar * (z(+) - z(-))
 * 	[-pi/2, pi/2] | phi (no constraint needed, but |phi| > pi/2 is handled by (zStar => 1 - zStar))
 * 
 *  This gives:
 * 
 * 	u_b = uFrac * usum
 * 	u_c = (1-uFrac) * usum
 * 	z = z(-) + zStar * (z(+) - z(-))
*/
class ShowerParticle : public ShapedJet
{
	public:
		
		//! @brief An exception thrown when a bad particle address is given to LocateParticle.
		class address_error : public std::runtime_error
		{
			public:
				address_error(std::string const& what_in):runtime_error(what_in) {}
		};
	
	private: 
		ShowerParticle* mother; // Currently not used at all, but keep around for versatility
		ShowerParticle* b; //! @brief Daughter b
		ShowerParticle* c; //! @brief Daughter c
		//~ std::shared_ptr<ShowerParticle> b; //! @brief Daughter b
		//~ std::shared_ptr<ShowerParticle> c; //! @brief Daughter c
		
		// I originally switched to shared_ptr for b and c because Cython needs a 
		// nullary constructor to do its automatic code.
		// It then sets the null-constructed object using copy assignment from an r-value.
		// For a binary tree, this shallow copy is problematic, 
		// because the original tree will be destroyed along with the r-value, 
		// so when the l-value is destroyed it will re-delete memory. A problem.
		// Using shared_ptr fixes this problem.
		// However, a more useful fix was deleting copy assignment and 
		// declaring a move assignment which swaps the pointers.
	
		/*! @brief The polarization is a unit-vector orthogonal to the splitting plane.
		 * 
		 *  pol is a property of the daughters, not the mother which splits.
		*/
		vec3_t pol;
		
		/*! @brief The supplied splitting parameters, saved for sanity checks.
		 *  
		 *  This container is empty until this particle is Split().
		*/ 
		std::vector<real_t> splittingParams;
		
		//! @brief A flag to record if energy is not exactly conserved after splitting.
		bool inexact;
				
		//! @brief A ShowerParticle by its \ref mother when she is instructed to Split().
		ShowerParticle(ShowerParticle* mother_in, 
			vec3_t const& p3_in, real_t const mass_in, vec3_t const& pol_in, 
			std::vector<bool>&& address_in);
		
		/*! @brief Split this particle using the splitting parameters in the range [begin, end).
		 *  
		 *  splitting parameters are assumed to be in this format (phi is optional)
		 *      splittingParams = {uSum, ubFrac, zStar, [phi]}
		*/		
		void Split(param_iter_t const param_begin, param_iter_t param_end);
		
		/*! @brief Given b's 3-momentum, b's mass fraction, 
		 *  and the total mass fraction of the daughters,
		 *  construct both daughters in a way that guarantees momentum conservation
		 *  (and for which energy should be conserved, albeit with rounding error).
		 * 
		 *  If energy is not exactly conserved, set the \ref inexact flag
		 *  (but do nothing more extreme; no exceptions are thrown, no assertions are made).
		*/ 
		void MakeDaughters(vec3_t const& p3_b, 
			real_t const u_b, real_t const uSum, vec3_t const& newPol);
			
		std::vector<bool> DaughterAddress(bool const which) const;
		
		/*! @brief Recursively append, to the existing vector,
		 *  copies of any final-state jets from this particle or its descendants.
		 * 
		 *  Only "leaves" (\ref isLeaf) are appended. 
		 *  Used by GetJets() to find all final-state jets.
		*/ 
		void AppendJets(std::vector<ShapedJet>& existing);
		
		//! @brief The CM frame momentum function.
		static real_t Delta2(real_t const uSum, real_t const uDiff);
		
		//! @brief Convert an address (see LocateParticle) to a string.
		static std::string AddressToString(std::vector<bool> const& address);
		
		/*! @brief Construct an address_error reporting that an address doesn't exist, 
		 *  and at what level/index in the address this occurs.
		*/  
		static address_error NoSuchAddress(std::vector<bool> const& address, size_t const level);
		
		/*! @brief Construct an address_error reporting that an address has already split,
		 *  and thus cannot be split again.
		*/
		static address_error AddressAlreadySplit(std::vector<bool> const& address);
		
		/*! @brief The energy lost during the splitting.
		 * 
		 *  \note This is unsafe because it de-references \ref b and \ref c without 
		 *  checking for their validity.
		 */ 
		real_t EnergyLoss_unsafe() // 
		{
			return (b->p4.x0 + c->p4.x0) - p4.x0;
		}
		
	public:
		ShowerParticle(): mother(nullptr), b(nullptr), c(nullptr) {}
		/*! @brief Build the shower. This is the root/original particle.
		 * 
		 *  The shower is constructed in the CM frame of the root,
		 *  with a mass of exactly one (scale as necessary).
		 *  A boost can be applied to the shower later (if one is needed).
		 *  
		 *  The root splitting occurs along the z-axis,
		 *  with the initial polarization along the x-axis.
		 *  A rotation can be applied to the shower later (if one is needed).
		 * 
		 *  Given this paradigm, the parameters \em must follow this format 
		 *  	\p params = 
		 * 		{uSum_0, ubFrac_0, 		<== 2 params for the root particle (splitting_0)
		 * 		 uSum_1, ubFrac_1, zStar_1,		<== 3 params for splitting_1
		 * 		 ...
		 * 		 uSum_i, ubFrac_i, zStar_i, phi_i, ...}		<== 4 params for all other splittings
		 * 
		 *  The root particle needs no address, but all additional splittings do:
		 *  	\p addresses = {addr_1, ..., addr_i, ...} 	
		*/ 
		ShowerParticle(std::vector<real_t> const& params, 
			std::vector<std::vector<bool>> const& addresses = {});
			
		// Delete the copy assigment 
		ShowerParticle& operator=(ShowerParticle const& orig) = delete;
		ShowerParticle& operator=(ShowerParticle&& orig);
		
		//! @brief Recursively destroy this particle and all descendants.
		~ShowerParticle();
	
		//! @brief In the shower tree, a branch has daughters (is has split).
		bool isBranch();
		
		//! @brief In the shower tree, a leaf has no daughters (it has not split).
		inline bool isLeaf() {return not isBranch();}
		
		/*! @brief \p true if this particle's energy was not exactly conserved when it split
		 *  (always \p false if isLeaf).
		*/  
		inline bool isInexact() {return inexact;}
		
		//! @brief \p true if this particle, or any descendants, isInexact
		bool isShowerInexact();
		
		/*! @brief Locate a particle in the shower, given its address relative to this particle.
		 *  
		 *  Each \p bool in address instructs which way to descend into the shower
		 *  (\p false = choose b-daughter, \p true = choose c-daughter).
		 * 
		 * 	address = {}              ==> this particle
		 *  	address = {False}         ==> this particle's b-daughter
		 *  	address = {True}          ==> this particle's c-daughter
		 * 	address = {True, False}   ==> the b-daughter of this particle's c-daughter		 
		*/		
		ShowerParticle& LocateParticle(std::vector<bool> const& address);
		
		//! @brief From this particle, copy all final-state descendants as ShapedJets.
		std::vector<ShapedJet> GetJets();
		
		//! @brief Return the energy lost when this particle split (0 if isLeaf).
		real_t EnergyLoss();
		
		//! @brief Return the sum of 4-momenta for the jets returned by this->GetJets.
		vec4_t Total_p4();
		
		/*! @brief From this particle, descend into the shower and 
		 *  add up the (absolute) energy lost by all splittings.
		*/
		real_t Total_absElost(real_t const absElost_in = real_t(0));
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
	public:
		using vec3_t = Jet::vec3_t;
		using vec4_t = Jet::vec4_t;
		using real_t = Jet::real_t;
		//~ typedef SpectralPower::PhatF PhatF;
		
		class JetParticle_Cache
		{
			friend class NjetModel;
			
			std::vector<ShapedJet> jetVec;
			std::vector<std::vector<real_t>> rho_jet;				
			
			JetParticle_Cache(NjetModel const& modeler,
				std::vector<ShapedJet> const& jetVec_in,
				size_t const lMax, real_t const jetShapeGranularity);
				
			public:
				JetParticle_Cache() {} //! @brief a public nullary ctor for Cython
				size_t lMax() const {return rho_jet.front().size();}
		};
	
	private:
		mutable pqRand::engine gen;
		
		//! @brief Do the work in the i-loop of H_l
		static std::vector<std::vector<real_t>> DoIncrements_jet_i(
			size_t const i, size_t const lMax,
			std::vector<ShapedJet> const& jetVec,
			//~ kdp::MutexCount<size_t>& kShared, 
			size_t const numIncrements,
			std::string const& generator_seed, 
			bool const onlySelf);
			
		std::vector<std::vector<real_t>> rho_j_l(
			size_t const i, size_t const lMax,
			real_t const jetShapeGranularity, real_t const Etot,
			std::vector<ShapedJet> const& jetVec_sorted,
			bool const onlySelf) const;
			
		static std::vector<ShapedJet> SortBy_E(std::vector<ShapedJet>);
		static real_t Total_E(std::vector<ShapedJet>);
		
	public:
		NjetModel(QSettings const& settings);
		NjetModel(std::string const& iniFileName = "NjetModel.conf");
		NjetModel();
		~NjetModel();
		
		JetParticle_Cache Make_JetParticle_Cache(std::vector<ShapedJet> const& jetVec,
				size_t const lMax, real_t const jetShapeGranularity) const;
		
		/*! @brief Given a vector of jets (in random order), 
		 *  return H_l from (l = 1) to (l = lMax).
		 * 
		 *  \param lMax 	the maximum \p returned
		 *  \param jetShapeGranularity	approximately how many random numbers will be drawn  
		*/ 
		std::vector<real_t> H_l(std::vector<ShapedJet> const& jetVec_unsorted, 
			size_t const lMax, real_t const jetShapeGranularity) const;
		
		// The power spectrum from multiplying rho_jets * rho_particles	
		std::vector<real_t> H_l_JetParticle(JetParticle_Cache const& cache, 
			std::vector<SpectralPower::PhatF> const& particles, 
			vec3_t const& axis, real_t const angle) const;
			
		static std::pair<real_t, real_t> CosSin(vec3_t const&, vec3_t const&);
					
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
