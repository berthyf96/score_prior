"""Utility functions for working with EHT data."""
import h5py
import ehtim as eh
import ehtim.const_def as ehc
import numpy as np
from scipy.linalg import block_diag

RA = 12.513728717168174  # right ascension of M87 observation
DEC = 12.39112323919932  # declination of M87 observation


def cphase_diag_info(obs):
    out = obs.c_phases_diag()
    # Form transformation matrix.
    dcp_list = []
    dcperr_list = []
    tform_matrix_list = []
    for d in out:
      dcp_list.extend(list(d[0]['cphase']))
      dcperr_list.extend(list(d[0]['sigmacp']))
      tform_matrix = d[4]['tform_matrix']
      tform_matrix_list.append(tform_matrix)

    dcp = np.array(dcp_list)
    dcperr = np.array(dcperr_list)

    # Make full transformation matrix of diagonalized closure phases.
    T = block_diag(*tform_matrix_list)

    # Check that data is the same as with official EHT code.
    obs_ref = obs.copy()
    obs_ref.add_cphase_diag()
    assert np.allclose(dcp, obs_ref.cphase_diag['cphase'])
    assert np.allclose(dcperr, obs_ref.cphase_diag['sigmacp'])

    return dcp, dcperr, T


def logcamp_diag_info(obs):
    out = obs.c_log_amplitudes_diag()
    # Form transformation matrix.
    logcamp_list = []
    logcamperr_list = []
    tform_matrix_list = []
    for d in out:
      logcamp_list.extend(list(d[0]['camp']))
      logcamperr_list.extend(list(d[0]['sigmaca']))
      tform_matrix = d[4]['tform_matrix']
      tform_matrix_list.append(tform_matrix)

    logcamp = np.array(logcamp_list)
    logcamperr = np.array(logcamperr_list)

    # Make full transformation matrix of diagonalized closure phases.
    T = block_diag(*tform_matrix_list)

    # Check that data is the same as with official EHT code.
    obs_ref = obs.copy()
    obs_ref.add_logcamp_diag()
    assert np.allclose(logcamp, obs_ref.logcamp_diag['camp'])
    assert np.allclose(logcamperr, obs_ref.logcamp_diag['sigmaca'])

    return logcamp, logcamperr, T


def estimate_flux_multiplier(obs: eh.obsdata.Obsdata,
                             npix: int,
                             fov: float = 128 * ehc.RADPERUAS,
                             zbl: float = 0.6,
                             prior_fwhm: float = 40 * ehc.RADPERUAS):
  """Estimate flux multiplier so that image pixels are in [0, 1].
  
  Args:
    obs: Obsdata, used only for metadata about the observation.
    npix: Number of pixels in each direction.
    fov: Field of view (in radians) of the image.
    zbl: Total compact flux density (Jy).
    prior_fwhm: FWHM (in radians) of Gaussian blob.
  """
  gaussprior = eh.image.make_square(obs, npix, fov)
  gaussprior = gaussprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
  return 1 / gaussprior.ivec.max()


def M87_systematic_noise(add_LM: bool = True):
  # Specify the SEFD error budget
  # (reported in First M87 Event Horizon Telescope Results III: Data Processing and Calibration)
  SEFD_error_budget = {'AA':0.10,
                      'AP':0.11,
                      'AZ':0.07,
                      'LM':0.22,
                      'PV':0.10,
                      'SM':0.15,
                      'JC':0.14,
                      'SP':0.07}

  # Add systematic noise tolerance for amplitude a-priori calibration errors
  # Start with the SEFD noise (but need sqrt)
  # then rescale to ensure that final results respect the stated error budget
  systematic_noise = SEFD_error_budget.copy()
  for key in systematic_noise.keys():
      systematic_noise[key] = ((1.0+systematic_noise[key])**0.5 - 1.0) * 0.25

  if add_LM:
    # Extra noise added for the LMT, which has much more variability than the a-priori error budget
    systematic_noise['LM'] += 0.15

  return systematic_noise


def combine_obs(obs1, obs2):
  # Add a slight offset to avoid mixed closure products
  obs2.data['time'] += 0.00001

  # concatenate the observations into a single observation object
  obs = obs1.copy()
  obs.data = np.concatenate([obs1.data, obs2.data])
  return obs


def rescale_zerobaseline(obs, totflux, orig_totflux, uv_max):
  """Rescale short baselines to excise contributions from extended flux.

  Setting totflux < orig_totflux assumes there is an extended constant flux component
  of orig_totflux - totflux [Jy].
  """
  multiplier = totflux / orig_totflux
  for j in range(len(obs.data)):
    if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 >= uv_max: continue
    for field in ['vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma']:
      obs.data[field][j] *= multiplier


def preprocess_obs(obs_orig, zbl=0.6, uv_zblcut=0.1e9, rescale_zbl=True):
  """Preprocess obsdata as in eht-imaging.

  Args:
    obs_orig: Obsdata, the original observation.
    zbl: Total compact flux density [Jy].
    sys_noise: Fractional systematic noise.
    uv_zblcut: u-v distance that separates the inter-site from intra-site baselines.
    reverse_taper_uas: Finest resolution of reconstructed features [uas].
  """
  obs = obs_orig.copy()

  # Estimate the total flux density from the AA -- AP zero baseline.
  zbl_tot = np.median(obs.unpack_bl('AA','AP','amp')['amp'])
  if zbl > zbl_tot:
    print('Warning: Specified total compact flux density ' +
          'exceeds total flux density measured on AA-AP!')

  # Flag out sites in the obs.tarr table with no measurements
  allsites = set(obs.unpack(['t1'])['t1'])|set(obs.unpack(['t2'])['t2'])
  obs.tarr = obs.tarr[[o in allsites for o in obs.tarr['site']]]
  obs = eh.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obs.data, obs.tarr,
                          source=obs.source, mjd=obs.mjd,
                          ampcal=obs.ampcal, phasecal=obs.phasecal)

  # Rescale short baselines to excise contributions from extended flux.
  if zbl != zbl_tot and rescale_zbl:
    print('RESCALING ZERO BASELINE')
    rescale_zerobaseline(obs, zbl, zbl_tot, uv_zblcut)

  # Order the stations by SNR.
  # This will create a minimal set of closure quantities with
  # the highest SNR and smallest covariance.
  obs.reorder_tarr_snr()

  return obs


def precalibrate_obs(obs_orig, npix, fov, sys_noise=0.0,
                     reverse_taper_uas=0.0, ttype='nfft'):
  """Pre-calibrate obsdata as in eht-imaging (happens after preprocessing)."""
  obs = obs_orig.copy()

  # Reverse taper the observation: this enforces a maximum resolution on
  # reconstructed features.
  if reverse_taper_uas > 0:
    obs = obs.reverse_taper(reverse_taper_uas * eh.RADPERUAS)

  # Add non-closing systematic noise to the observation.
  obs = obs.add_fractional_noise(sys_noise)

  # Make a copy of the initial data (before any self-calibration but after the taper)
  obs_sc_init = obs.copy()

  # Self-calibrate the LMT to a Gaussian model
  # (Refer to Section 4's "Pre-Imaging Considerations")
  print("Self-calibrating the LMT to a Gaussian model for LMT-SMT...")

  obs_LMT = obs_sc_init.flag_uvdist(uv_max=2e9) # only consider the short baselines (LMT-SMT)
  if reverse_taper_uas > 0:
    # start with original data that had no reverse taper applied.
    # Re-taper, if necessary
    obs_LMT = obs_LMT.taper(reverse_taper_uas * ehc.RADPERUAS)

  # Make a Gaussian image that would result in the LMT-SMT baseline visibility amplitude
  # as estimated in Section 4's "Pre-Imaging Considerations".
  # This is achieved with a Gaussian of size 60 microarcseconds and total flux of 0.6 Jy
  gausspriorLMT = eh.image.make_square(obs, npix, fov)
  gausspriorLMT = gausspriorLMT.add_gauss(
    0.6,
    (60.0 * eh.RADPERUAS, 60.0 * eh.RADPERUAS, 0, 0, 0))

  # Self-calibrate the LMT visibilities to the gausspriorLMT image
  # to enforce the estimated LMT-SMT visibility amplitude
  caltab = eh.selfcal(obs_LMT, gausspriorLMT, sites=['LM'], gain_tol=1.0,
                      method='both', ttype=ttype, caltable=True)

  # Supply the calibration solution to the full (and potentially tapered) dataset
  obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

  return obs


def load_im_hdf5(filename):
    """Read in an image from an hdf5 file.
       Args:
            filename (str): path to input hdf5 file
       Returns:
            (Image): loaded image object
    """

    # Load information from hdf5 file

    hfp = h5py.File(filename,'r')
    dsource = hfp['header']['dsource'][()]          # distance to source in cm
    jyscale = hfp['header']['scale'][()]            # convert cgs intensity -> Jy flux density
    rf = hfp['header']['freqcgs'][()]               # in cgs
    tunit = hfp['header']['units']['T_unit'][()]    # in seconds
    lunit = hfp['header']['units']['L_unit'][()]    # in cm
    DX = hfp['header']['camera']['dx'][()]          # in GM/c^2
    nx = hfp['header']['camera']['nx'][()]          # width in pixels
    time = hfp['header']['t'][()] * tunit / 3600.       # time in hours
    if 'pol' in hfp:
        poldat = np.copy(hfp['pol'])[:, :, :4]            # NX,NY,{I,Q,U,V}
    else: # unpolarized data only
        unpoldat = np.copy(hfp['unpol'])                # NX,NY
        poldat = np.zeros(list(unpoldat.shape)+[4])
        poldat[:,:,0] = unpoldat
    hfp.close()

    # Correct image orientation
    # unpoldat = np.flip(unpoldat.transpose((1, 0)), axis=0)
    poldat = np.flip(poldat.transpose((1, 0, 2)), axis=0)

    # Make a guess at the source based on distance and optionally fall back on mass
    src = ehc.SOURCE_DEFAULT
    if dsource > 4.e25 and dsource < 6.2e25:
        src = "M87"
    elif dsource > 2.45e22 and dsource < 2.6e22:
        src = "SgrA"

    # Fill in information according to the source
    ra = ehc.RA_DEFAULT
    dec = ehc.DEC_DEFAULT
    if src == "SgrA":
        ra = 17.76112247
        dec = -28.992189444
    elif src == "M87":
        ra = 187.70593075
        dec = 12.391123306

    # Process image to set proper dimensions
    fovmuas = DX / dsource * lunit * 2.06265e11
    psize_x = ehc.RADPERUAS * fovmuas / nx

    Iim = poldat[:, :, 0] * jyscale
    Qim = poldat[:, :, 1] * jyscale
    Uim = poldat[:, :, 2] * jyscale
    Vim = poldat[:, :, 3] * jyscale

    outim = eh.image.Image(Iim, psize_x, ra, dec, rf=rf, source=src,
                              polrep='stokes', pol_prim='I', time=time)
    outim.add_qu(Qim, Uim)
    outim.add_v(Vim)

    return outim
