#ifndef N_JET_MODEL
#define N_JET_MODEL

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerJets.hpp"
#include "RecursiveLegendre.hpp"
#include "pqRand/pqRand.hpp"
#include <vector>
#include <memory> // ShowerParticle
#include "fastjet/PseudoJet.hh"
#include "ShapeFunction.hpp"
#include <QtCore/QSettings>

#include <Pythia8/Pythia.h>

/*! @file NjetModel.hpp
 *  @brief Defines jet classes and a binary splitting tree to define jets
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/

/*! @brief A 4-momenta with a cached mass.
 * 
 *  This is useful when the object is significantly boosted, 
 *  since calculating it's mass from E**2 - p**2 loses precision.
*/
struct Jet
{
	using real_t = PowerJets::real_t;
	using vec3_t = PowerJets::vec3_t; //!< @brief The 3-vector type
	using vec4_t = PowerJets::vec4_t; //!< @brief The 4-vector type
	
	vec4_t p4; //!> @brief The 4-momentum
	
	/*! @brief The mass is cached to avoid re-computation (with rounding error).
	 * 
	 *  \warning The mass is not actively managed after construction. 
	 *  If you operate on p4, you must recompute mass.
	*/
	real_t mass; 
	
	Jet():p4(), mass(0) {}
	
	//! @brief Dummy boolean means don't zero-initialize the 4-momentum
	explicit Jet(bool):p4(false) {}
	
	explicit Jet(vec4_t const& p4_in):
		p4(p4_in), mass(p4.Length()) {}	
	
	//! @brief This is the main ctor, which all others should call.
	explicit Jet(vec3_t const& p3_in, real_t const w0, kdp::Vec4from2 const w0type);
			
	//! @brief The interface we expect to use from inside a Cython loop
	explicit Jet(real_t const x1, real_t const x2, real_t const x3, 
		real_t const w0, kdp::Vec4from2 const w0type);
		
	Jet(fastjet::PseudoJet const& pj);
	
	Jet(Pythia8::Particle const& particle);
	
	//! @brief Rotate about \p axis by \p angle (in radians)
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
 *  ShapedJets are designed to be constructed from a tree of ShowerParticle's;
 *  they are essentially ShowerParticle's stripped of their binary tree pointers.
 *  
 *  At the moment the jet's CM shape is always isotropic,
 *  but the interface allows more sophisticated (though azimuthally symmetric) shapes 
 *  without altering the API.
*/
class ShapedJet : public Jet
{
	public:
		/*! @brief The address of the jet in the ShowerParticle splitting tree.
		 * 
		 *  The address is described by ShowerParticle. 
		 *  It is kept in ShapedJet for simplicity of the Cython code, 
		 *  but it really shouldn't exist in ShapedJet at all.
		*/ 
		std::vector<bool> address;
		mutable h_Boost shape;
		
		using Jet::Rotate;		
	
		//! @brief Random samples from the jet shape are generated in increments of this size
		static constexpr size_t incrementSize = size_t(1) << 8; // 512, not too large, not too small
		using incrementArray_t = std::array<real_t, incrementSize>;
		
		//! @brief The container to pass parameters defining non-isotropic shapes
		using param_iter_t = std::vector<real_t>::const_iterator;
		
			GCC_IGNORE_PUSH(-Wunused-parameter)
		/*! @brief The azimuthally symmetric jet shape function.
		 *
		 *  A normalized PDF for \p z_CM = cos(theta_CM), the CM-frame polar angle
		 *  (with the +z direction defined by jet's lab-frame 3-momentum).
		*/
		real_t h_CM(real_t const z_CM) const {return real_t(0.5);}
			GCC_IGNORE_POP
		
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
		
		ShapedJet(): Jet(), shape(p4.p(), real_t(1)) {} // A minimal nullary constructor so Cython can use push_back
		
		~ShapedJet() {}
				
		//! @brief The don't initialize constructor
		explicit ShapedJet(bool): Jet(false), shape() {}
		
			GCC_IGNORE_PUSH(-Wunused-parameter)
		ShapedJet(vec3_t const& p3_in, real_t const w0, kdp::Vec4from2 const w0type,
			std::vector<bool> address_in = std::vector<bool>(),
			std::vector<real_t> const& shapeParams = {}):
		Jet(p3_in, w0, w0type),
		address(std::move(address_in)),
		shape(p4.p(), mass) {}
				
		//! @brief The interface we expect to use from inside a Cython loop
		ShapedJet(real_t const x1, real_t const x2, real_t const x3, 
			real_t const w0, kdp::Vec4from2 const w0type, 
			std::vector<bool> address_in = std::vector<bool>(),
			std::vector<real_t> const& shapeParams = {}):
		Jet(x1, x2, x3, w0, w0type),
		address(std::move(address_in)),
		shape(p4.p(), mass) {}
			GCC_IGNORE_POP
		
		// We assume that the shape of the jet will be initialized later, 
		// after the jet's 4-vector and mass have been defined.
		void SetShape(param_iter_t const shapeParam_begin, param_iter_t const shapeParam_end);
		
		//~ std::vector<real_t> OnAxis(size_t const lMax) const {return shape.hl_Vec(lMax);}
		
		// cython does not support
		//~ static bool Sort_by_Mass(ShapedJet const& left, ShapedJet const& right)
		//~ {
			//~ return left.mass > right.mass;
		//~ }
		
		//! @brief Sort ShapedJet's by their mass
		bool operator < (ShapedJet const& that) const;
};

GCC_IGNORE_PUSH(-Wpadded)

/*! @brief An extension of ShapedJet which manages a particle shower built from 
 *  mother particles splitting to two daughters (\f$ a \to b \, c \f$).
 *  Each ShowerParticle is a node in a binary tree.
 * 
 *  \section params Splitting parameters
 * 
 *  When each mother is split, she has some energy \f$ E_a \f$  and mass \f$ m_a \f$ (both non-negative). 
 *  The \em b-c splitting plane is defined by a polarization vector:
 *  \f[ \hat{\epsilon} \equiv \frac{\hat{p}_b \times \hat{p}_c}{|\hat{p}_b \times \hat{p}_c|}\f]
 *  The mother's splitting is determined by 4 parameters (the 4 d.o.f for particle \em b, 
 *  with particle \em c defined by conservation of momentum):
 *   
 *   - \em z : the splitting fraction \f$ 0 \le z \le 1;\; E_b = z\, E_a,\; E_c = (1-z) E_a \f$
 * 
 *   - \f$ u_b \f$ : \em b's (non-negative) mass fraction; \f$ m_b = u_b\, m_a \f$
 * 
 *   - \f$ u_c \f$ : \em c's (non-negative) mass fraction; \f$ m_c = u_c\, m_a \f$
 * 
 *   - \f$ \phi \f$ : the angle of the active, right-handed rotation 
 *     about the mother's direction of travel \f$ \hat{p}_a \f$ that takes the mother's polarization 
 *     \f$ \hat{\epsilon}_a \f$ (which defined the splitting plane that spawned the mother)
 *     to her daughters' polarization \f$ \hat{\epsilon}_{bc} \f$ (defining the \em b-c splitting plane).
 * 	
 *  There are a number of constraints which are needed for each splitting.
 *  The daughters' total mass must conserve energy
 * 
 * 	\f[ u_b + u_c \le 1 \f]
 * 
 *  Also, \em z cannot arbitrarily satisfy \f$ p^\mu \f$ conservation;
 *  it must satisfy  \f$ z_- \le z \le z_+ \f$, where:
 * 
 *  \f[z_\pm = \frac{1}{2}\left(1 + (q_b^2 - q_c^2) \pm \beta_a
 *     \sqrt{(1  - (q_b + q_c)^2)(1 - (q_b - q_c)^2)}\right) \f]
 * 
 *  We expect that the shower parameters will be determined from a 
 *  non-linear minimization fit. In general, while it is easy to 
 *  set "bounds" on parameters (i.e. \c param_i must exist in some domain), 
 *  it is non-trivial to enforce constraints (i.e. some equality of inequality 
 *  which depends on multiple parameters must be satisfied).
 *  To avoid constraints, we can build the constraints into the fitting parameters
 *  (so that, if the constraints are non-linear, the non-linearity is 
 *  baked into the fit parameters). We choose the following fit parameters:
 * 
 *   - \f$ u_{bc} \equiv \frac{m_b + m_c}{m_a} \in [0,1] \f$
 * 
 *   - \f$ u_b^* \equiv \frac{m_b}{m_b + m_c} \in [0,1] \f$
 * 
 *   - \f$ z^* \equiv \frac{z - z_-}{z_+ - z_-} \in [0,1] \f$
 * 
 *   - \f$ \phi \in [-\pi/2, \pi/2] \f$ 
 *     (no constraint needed, but \f$ |\phi| > \pi/2 \f$ is handled by \f$ z^* \to 1 - z^* \f$)
 * 
 *  This gives:
 *  
 *   - \f$ u_b = u_b^*\, u_{bc} \f$
 *   - \f$ u_c = (1-u_b^*) u_{bc} \f$ 
 *   - \f$ z = z_- + z^* \, (z^+ - z_-) \f$
 * 
 *  Each node uses these four bounded d.o.f (see ShowerParticle 
 *  for a discussion of the root node and the tree orientation).
 * 
 *  \section address Node address
 * 
 *  The binary tree is composed of nodes, but a node can be a branch or a leaf. 
 *  A branch is a node that has already split; it cannot split again. 
 *  Only leaves can be split, but a tree can have many leaves. 
 *  Hence, specifying a new splitting takes more than the four kinematic parameters; 
 *  we also need to identify which leaf is splitting. 
 *  This is accomplished via the node "address".
 *  For simplicity, the address is built from directions for 
 *  navigating the tree (see \ref LocateParticle): 
 * 
 *   - \c false means "follow daughter b".
 *   - \c true means "follow daughter c".
 * 
 *  All the directions to find a node, starting from the root node, 
 *  are wrapped into a <tt> std::vector<bool> </tt>.
 *  Hence, an empty vector is the address of the root node. 
 *  Of course, just because an address can be written doesn't make it valid. 
 *  For safety, \ref LocateParticle will throw an exception when an address is invalid.
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
		ShowerParticle* mother; //!< @brief Currently not used, but keep around for versatility
		ShowerParticle* b; //!< @brief Daughter b
		ShowerParticle* c; //!< @brief Daughter c
		//~ std::shared_ptr<ShowerParticle> b; //!< @brief Daughter b
		//~ std::shared_ptr<ShowerParticle> c; //!< @brief Daughter c
		
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
		 *  \f[ \hat{\epsilon} \equiv \frac{\hat{p}_b \times \hat{p}_c}{|\hat{p}_b \times \hat{p}_c|}\f]
		 *  \ref pol is a property of the daughters, not the mother which splits.
		*/
		vec3_t pol;
		
		/*! @brief The supplied splitting parameters, saved for sanity checks.
		 *  
		 *  This container is empty until this particle is Split().
		*/ 
		std::vector<real_t> splittingParams;
		
		/*! @brief A flag to record if energy is not exactly conserved during splitting.
		 * 
		 *  \note Momentum is conserved by construction of \ref MakeDaughters
		*/ 
		bool inexact;
				
		//! @brief A ShowerParticle spawned by a \ref mother who was instructed to Split().
		ShowerParticle(ShowerParticle* mother_in, 
			vec3_t const& p3_in, real_t const mass_in, vec3_t const& pol_in, 
			std::vector<bool>&& address_in);
		
		/*! @brief Split this particle using the splitting parameters in the range [begin, end).
		 *  
		 *  Splitting parameters are assumed to be in this format (phi is optional) \verbatim
		    splittingParams = {u_bc, u_b_star, z_star, [phi]}
		    \endverbatim
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
			real_t const u_b, real_t const u_bc, vec3_t const& newPol);
		
		//! @brief The address of the daughter (\c false => b, \c true => c)
		std::vector<bool> DaughterAddress(bool const which) const;
		
		/*! @brief Recursively append, to the existing vector,
		 *  copies of any final-state jets from this particle or its descendants.
		 * 
		 *  Only "leaves" (\ref isLeaf) are appended. 
		 *  Used by GetJets() to find all final-state jets.
		*/ 
		void AppendJets(std::vector<ShapedJet>& existing) const;
		
		//! @brief The CM frame momentum function.
		static real_t Delta2(real_t const sum, real_t const diff);
		
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
		real_t EnergyLoss_unsafe() const // 
		{
			return (b->p4.x0 + c->p4.x0) - p4.x0;
		}
		
	public:
		//! @brief A nullary constructor for Cython
		ShowerParticle(): mother(nullptr), b(nullptr), c(nullptr) {}
		
		/*! @brief Build a shower in the CM frame of this particle (the root particle). 
		 * 
		 *  The shower is constructed in the CM frame of the root,
		 *  with a mass of exactly one (scale energy as necessary).
		 *  A boost can be applied to the shower later (if one is needed).
		 *  Because the power spectrum is invariant to absolute orientation, 
		 *  the three degrees of freedom defining the absolute orientation 
		 *  are optional. We explain why in the next paragraph.
		 * 
		 *  Because the root node is defined in its CM frame,
		 *  the two root daughters are back-to-back, which makes their 
		 *  polarization vector (and hence polarization rotation angle) 
		 *  ill-defined (i.e., any vector in the 
		 *  orthogonal plane is orthogonal to both daughters).		 
		 *  Additionally, since \f$ \beta_a = 0 \f$,
		 *  \f$ z_+ = z_- \f$ and \f$ z^* \f$ is meaningless.
		 *  Yet these two degrees of freedom do not disappear, 
		 *  they merely transform into the two d.o.f. defining the 
		 *  \em axis of the two original daughters.
		 *  When one of these original daughter's splits, 
		 *  their non-zero speed releases \f$ and z^* \f$, 
		 *  but \f$ \phi \f$ remains ill-defined
		 *  because there is no existing polarization vector.
		 *  Thus, the absolute orientation of the splitting tree
		 *  requires defining these three d.o.f.
		 *  (two to define the root axis, and one to define the 
		 *  polarization of the first splitting, via its azimuthal position 
		 *  about the root axis).
		 * 
		 *  Recall that the power spectrum is invariant to absolute orientation, 
		 *  so defining the three orientation d.o.f. may not be necessary. 
		 *  Thus, we default to an arbitrary orientation of the root splitting:
		 *  the root axis is the \em z axis,  with daughter \em b going in the 
		    \f$ +\hat{z} \f$ direction, and the root polarization is the \em x axis.
		 *  If the three optional orientation d.o.f. are supplied, 
		 *  they record the following orientation variables:
		 *  
		 *   - \f$ \theta_0 \f$: The polar angle of root daughter \em b
		 *     (relative to the \f$ +\hat{z} \f$).
		 * 
		 *   - \f$ \phi_0 \f$: The right-handed azimuthal angle of root daughter \em b
		 *     (relative to the \f$ +\hat{x} \f$).
		 *   
		 *   - \f$ \omega_0 \f$: The rotation of the root polarization 
		 *     about the daughter \em b's axis. 
		 *     The first two d.o.f. define an active, right-handed rotation which takes 
		 *     \f$ (0,\, 0,\, 1) \f$ to 
		 *     \f$ (\sin\theta_0 \cos\phi_0,\, \sin\theta_0 \sin\phi_0, \cos\theta_0)\f$, 
		 *     with a post-rotation about the new axis of \f$ \omega_0 \f$. 
		 * 
		 *  If there is a second splitting, regardless of whether it is 
		 *  root daughter \em b or \em c, it will use the root polarization
		 *  with \f$ \phi = 0 \f$ (i.e., the plane of the second splitting 
		 *  is defined by the root polarization, so \f$ \omega_0 \f$ orients this plane).
		 * 
		 *  Given this paradigm, the root splitting needs only two kinematic parameters, 
		 *  and the second splitting only three kinematic parameters. 
		 *  All additional nodes need the full four parameters.
		 *  
		 *  \param params_kinematic
		 *  The params specifying a k-node tree, with either
		 *  2 or 5 + 4(k-2) parameters
		 *  \verbatim
		    {u_bc_0, u_b_star_0,            [ 2 params for the root splitting ]
		     u_bc_1, u_b_star_1, z_star_1,  [ 3 params for splitting 1 ]
		     ...                          [ 4 params for all other splittings ]
		     u_bc_{k-1}, u_b_star_{k-1}, z_star_{k-1}, phi_{k-1}}   
		    \endverbatim
		 *
		 *  \param addresses 
		 *  A vector of addresses to specify the location of all nodes after the root node
		 *  (i.e. the root particle needs no address).
		 * 
		 *  \param params_orientation
		 *  The three orientation parameters described above: \verbatim
		    {theta_0, phi_0, omega_0}. \endverbatim
		 *  This vector shall either be empty or contain exactly three parameters.
		 * 
		 *  \throw Throws \c std::invalid_argument if \p params_kinematic 
		 *  does not have the correct length for a k-node tree.
		 * 
		 *  \throw Throws \c std::invalid_argument if \p addresses.size()
		 *  is not k-1, given the k-node tree specified by \p params_kinematic.
		 * 
		 *  \throw Throws \c std::invalid_argument if \p params_orientation.size()
		 *  is not zero or three.
		*/ 
		ShowerParticle(std::vector<real_t> const& params_kinematic, 
			std::vector<std::vector<bool>> const& addresses = {},
			std::vector<real_t> const& params_orientation = {});
			
		/*! @brief Build a shower in the CM frame of this particle (the root particle)
		 *  using one long list of parameters.
		 * 
		 *  Like the main constructor, but with [orientation, kinematic] 
		 *  parameters concatenated into the same list (orientation first).
		 * 
		 *  \note This function is less safe than the main constructor 
		 *  because it must \em detect the presence of orientation parameters 
		 *  and separate them out. It exists so that we can build a shower from 
		 *  the single list of parameters used by \c scipy.optimize.least_squares.
		 * 
		 *  \note We detect the presence of orientation parameters \em only when 
		 *  <tt> params_orientationKinematic.size() % 4 == 0 </tt>.
		*/ 
		static ShowerParticle FromParams_OrientationKinematic(std::vector<real_t> const& params_orientationKinematic, 
			std::vector<std::vector<bool>> const& addresses = {});
			
		/*! \defgroup OnlyMove 
		 *  @brief Nodes cannot be copied, they can only be moved.
		 * 
		 *  This prevents dealing with shallow/deep copies and pointer sharing/deletion.
		*/
		
		// We need a period after \ingroup, otherwise doxygen won't link them correctly
		ShowerParticle(ShowerParticle const& orig) = delete; //!< \ingroup OnlyMove .
		ShowerParticle& operator=(ShowerParticle const& orig) = delete; //!< \ingroup OnlyMove .
		ShowerParticle(ShowerParticle&& orig); //!< \ingroup OnlyMove .
		ShowerParticle& operator=(ShowerParticle&& orig); //!< \ingroup OnlyMove .
		
		//! @brief Recursively destroy this particle and all descendants.
		~ShowerParticle();
	
		//! @brief A "branch" in the shower tree has daughters (is has split).
		bool isBranch() const;
		
		//! @brief A "leaf" in the shower tree has no daughters (it has not split).
		//! This makes it a final-state particle.
		inline bool isLeaf() const {return not isBranch();}
		
		/*! @brief \p true if this particle's energy was not exactly conserved when it split
		 *  (always \p false if \ref isLeaf).
		*/  
		inline bool isInexact() const {return inexact;}
		
		//! @brief \p true if this particle, or any descendants, \ref isInexact
		bool isShowerInexact() const;
		
		/*! @brief Locate a particle in the shower, given its address relative to this particle.
		 *  
		 *  Each \p bool in address instructs which way to descend into the shower
		 *  (\p false = follow b-daughter, \p true = follow c-daughter).
		 * 
		 *  \verbatim
		    address = {}              ==> this particle
		    address = {False}         ==> this particle's b-daughter
		    address = {True}          ==> this particle's c-daughter
		    address = {True, False}   ==> the b-daughter of this particle's c-daughter
		    \endverbatim
		*/		
		ShowerParticle& LocateParticle(std::vector<bool> const& address);
		
		//! @brief Return the final-state (leaf) descendants of this particle as ShapedJets.
		std::vector<ShapedJet> GetJets() const;
		
		//! @brief Return the energy lost when this particle split (0 if isLeaf).
		real_t EnergyLoss() const;
		
		//! @brief Return the sum of 4-momenta for the jets returned by this->GetJets.
		//~ vec4_t Total_p4();
		
		/*! @brief From this particle, descend into the shower and 
		 *  add up the (absolute) energy lost by all splittings.
		*/
		real_t Total_absElost(real_t const absElost_in = real_t(0)) const;
};

GCC_IGNORE_POP

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
//~ class NjetModel
//~ {
	//~ public:
		//~ using vec3_t = Jet::vec3_t;
		//~ using vec4_t = Jet::vec4_t;
		//~ using real_t = Jet::real_t;
		//~ // typedef SpectralPower::PhatF PhatF;
		
		//~ class JetParticle_Cache
		//~ {
			//~ friend class NjetModel;
			
			//~ std::vector<ShapedJet> jetVec;
			//~ std::vector<std::vector<real_t>> rho_jet;				
			
			//~ JetParticle_Cache(NjetModel const& modeler,
				//~ std::vector<ShapedJet> const& jetVec_in,
				//~ size_t const lMax, real_t const jetShapeGranularity);
				
			//~ public:
				//~ JetParticle_Cache() {} //! @brief a public nullary ctor for Cython
				//~ size_t lMax() const {return rho_jet.front().size();}
		//~ };
	
	//~ private:
		//~ mutable pqRand::engine gen;
		
		//~ //! @brief Do the work in the i-loop of H_l
		//~ static std::vector<std::vector<real_t>> DoIncrements_jet_i(
			//~ size_t const i, size_t const lMax,
			//~ std::vector<ShapedJet> const& jetVec,
			//~ //kdp::MutexCount<size_t>& kShared, 
			//~ size_t const numIncrements,
			//~ std::string const& generator_seed, 
			//~ bool const onlySelf);
			
		//~ std::vector<std::vector<real_t>> rho_j_l(
			//~ size_t const i, size_t const lMax,
			//~ real_t const jetShapeGranularity, real_t const Etot,
			//~ std::vector<ShapedJet> const& jetVec_sorted,
			//~ bool const onlySelf) const;
			
		//~ static std::vector<ShapedJet> SortBy_E(std::vector<ShapedJet>);
		//~ static real_t Total_E(std::vector<ShapedJet>);
		
	//~ public:
		//~ NjetModel(QSettings const& settings);
		//~ NjetModel(std::string const& iniFileName = "NjetModel.conf");
		//~ NjetModel();
		//~ ~NjetModel();
		
		//~ JetParticle_Cache Make_JetParticle_Cache(std::vector<ShapedJet> const& jetVec,
				//~ size_t const lMax, real_t const jetShapeGranularity) const;
		
		//~ /*! @brief Given a vector of jets (in random order), 
		 //~ *  return H_l from (l = 1) to (l = lMax).
		 //~ * 
		 //~ *  \param lMax 	the maximum \p returned
		 //~ *  \param jetShapeGranularity	approximately how many random numbers will be drawn  
		//~ */ 
		//~ std::vector<real_t> H_l(std::vector<ShapedJet> const& jetVec_unsorted, 
			//~ size_t const lMax, real_t const jetShapeGranularity) const;
		
		//~ // The power spectrum from multiplying rho_jets * rho_particles	
		//~ std::vector<real_t> H_l_JetParticle(JetParticle_Cache const& cache, 
			//~ std::vector<SpectralPower::PhatF> const& particles, 
			//~ vec3_t const& axis, real_t const angle) const;
			
		//~ static std::pair<real_t, real_t> CosSin(vec3_t const&, vec3_t const&);
					
		//~ // static std::vector<vec4_t> GetJets(std::vector<real_t> const& jetParams);
		//~ // static std::vector<std::vector<real_t>> GetJetsPy(std::vector<real_t> const& jetParams);
				
		//~ // // Draw a unit vector isotropically from the unit sphere
		//~ // static vec3_t IsoVec3(pqRand::engine& gen);

		//~ // // Return n_request unit 3-vectors randomly but isotropically distributed, 
		//~ // // but which nonetheless sum to zero. Due to the balancing scheme, 
		//~ // // the number of vectors returned may occasionally be (n_request + 1)
		//~ // static std::vector<kdp::Vec3> IsoCM(size_t const n_request, 
			//~ // pqRand::engine& gen, real_t const tolerance = 1e-15);
//~ };

#endif
