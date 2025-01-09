"""
This module contains all neccessary files to compute the filters used
in the pyShearLab2D toolbox. Most of these files are taken from different
MATLAB toolboxes and were translated to Python. Credit is given in each
individual function.


Stefan Loock, February 2, 2017 [sloock@gwdg.de]
"""

from __future__ import division
import numpy as np
from scipy import signal as signal


try:
    import pyfftw

    fftlib = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
except ImportError:
    fftlib = np.fft


def MakeONFilter(Type, Par=1):
    """
    This is a rewrite of the original Matlab implementation of MakeONFilter.m
    from the WaveLab850 toolbox.

    MakeONFilter -- Generate Orthonormal QMF Filter for Wavelet Transform

    Usage:

        qmf = MakeONFilter(Type, Par)

    Inputs:

        Type:  string: 'Haar', 'Beylkin', 'Coiflet', 'Daubechies',
                        'Symmlet', 'Vaidyanathan', 'Battle'

    Outputs:

        qmf:    quadrature mirror filter

    Description

    The Haar filter (which could be considered a Daubechies-2) was the
    first wavelet, though not called as such, and is discontinuous.

    The Beylkin filter places roots for the frequency response function
    close to the Nyquist frequency on the real axis.

    The Coiflet filters are designed to give both the mother and father
    wavelets 2*Par vanishing moments; here Par may be one of 1,2,3,4 or 5.

    The Daubechies filters are minimal phase filters that generate wavelets
    which have a minimal support for a given number of vanishing moments.
    They are indexed by their length, Par, which may be one of
    4,6,8,10,12,14,16,18 or 20. The number of vanishing moments is par/2.

    Symmlets are also wavelets within a minimum size support for a given
    number of vanishing moments, but they are as symmetrical as possible,
    as opposed to the Daubechies filters which are highly asymmetrical.
    They are indexed by Par, which specifies the number of vanishing
    moments and is equal to half the size of the support. It ranges
    from 4 to 10.

    The Vaidyanathan filter gives an exact reconstruction, but does not
    satisfy any moment condition.  The filter has been optimized for
    speech coding.

    The Battle-Lemarie filter generate spline orthogonal wavelet basis.
    The parameter Par gives the degree of the spline. The number of
    vanishing moments is Par+1.

    See Also: FWT_PO, IWT_PO, FWT2_PO, IWT2_PO, WPAnalysis

    References: The books by Daubechies and Wickerhauser.

    Part of  WaveLab850 (http://www-stat.stanford.edu/~wavelab/)
    """
    if Type == "Haar":
        onFilter = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    if Type == "Beylkin":
        onFilter = np.array(
            [
                0.099305765374,
                0.424215360813,
                0.699825214057,
                0.449718251149,
                -0.110927598348,
                -0.264497231446,
                0.026900308804,
                0.155538731877,
                -0.017520746267,
                -0.088543630623,
                0.019679866044,
                0.042916387274,
                -0.017460408696,
                -0.014365807969,
                0.010040411845,
                0.001484234782,
                -0.002736031626,
                0.000640485329,
            ]
        )
    if Type == "Coiflet":
        if Par == 1:
            onFilter = np.array(
                [
                    0.038580777748,
                    -0.126969125396,
                    -0.077161555496,
                    0.607491641386,
                    0.745687558934,
                    0.226584265197,
                ]
            )
        elif Par == 2:
            onFilter = np.array(
                [
                    0.016387336463,
                    -0.041464936782,
                    -0.067372554722,
                    0.386110066823,
                    0.812723635450,
                    0.417005184424,
                    -0.076488599078,
                    -0.059434418646,
                    0.023680171947,
                    0.005611434819,
                    -0.001823208871,
                    -0.000720549445,
                ]
            )
        elif Par == 3:
            onFilter = np.array(
                [
                    -0.003793512864,
                    0.007782596426,
                    0.023452696142,
                    -0.065771911281,
                    -0.061123390003,
                    0.405176902410,
                    0.793777222626,
                    0.428483476378,
                    -0.071799821619,
                    -0.082301927106,
                    0.034555027573,
                    0.015880544864,
                    -0.009007976137,
                    -0.002574517688,
                    0.001117518771,
                    0.000466216960,
                    -0.000070983303,
                    -0.000034599773,
                ]
            )
        elif Par == 4:
            onFilter = np.array(
                [
                    0.000892313668,
                    -0.001629492013,
                    -0.007346166328,
                    0.016068943964,
                    0.026682300156,
                    -0.081266699680,
                    -0.056077313316,
                    0.415308407030,
                    0.782238930920,
                    0.434386056491,
                    -0.066627474263,
                    -0.096220442034,
                    0.039334427123,
                    0.025082261845,
                    -0.015211731527,
                    -0.005658286686,
                    0.003751436157,
                    0.001266561929,
                    -0.000589020757,
                    -0.000259974552,
                    0.000062339034,
                    0.000031229876,
                    -0.000003259680,
                    -0.000001784985,
                ]
            )
        elif Par == 5:
            onFilter = np.array(
                [
                    -0.000212080863,
                    0.000358589677,
                    0.002178236305,
                    -0.004159358782,
                    -0.010131117538,
                    0.023408156762,
                    0.028168029062,
                    -0.091920010549,
                    -0.052043163216,
                    0.421566206729,
                    0.774289603740,
                    0.437991626228,
                    -0.062035963906,
                    -0.105574208706,
                    0.041289208741,
                    0.032683574283,
                    -0.019761779012,
                    -0.009164231153,
                    0.006764185419,
                    0.002433373209,
                    -0.001662863769,
                    -0.000638131296,
                    0.000302259520,
                    0.000140541149,
                    -0.000041340484,
                    -0.000021315014,
                    0.000003734597,
                    0.000002063806,
                    -0.000000167408,
                    -0.000000095158,
                ]
            )
    if Type == "Daubechies":
        if Par == 4:
            onFilter = np.array(
                [0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551]
            )
        elif Par == 6:
            onFilter = np.array(
                [
                    0.332670552950,
                    0.806891509311,
                    0.459877502118,
                    -0.135011020010,
                    -0.085441273882,
                    0.035226291882,
                ]
            )
        elif Par == 8:
            onFilter = np.array(
                [
                    0.230377813309,
                    0.714846570553,
                    0.630880767930,
                    -0.027983769417,
                    -0.187034811719,
                    0.030841381836,
                    0.032883011667,
                    -0.010597401785,
                ]
            )
        elif Par == 10:
            onFilter = np.array(
                [
                    0.160102397974,
                    0.603829269797,
                    0.724308528438,
                    0.138428145901,
                    -0.242294887066,
                    -0.032244869585,
                    0.077571493840,
                    -0.006241490213,
                    -0.012580751999,
                    0.003335725285,
                ]
            )
        elif Par == 12:
            onFilter = np.array(
                [
                    0.111540743350,
                    0.494623890398,
                    0.751133908021,
                    0.315250351709,
                    -0.226264693965,
                    -0.129766867567,
                    0.097501605587,
                    0.027522865530,
                    -0.031582039317,
                    0.000553842201,
                    0.004777257511,
                    -0.001077301085,
                ]
            )
        elif Par == 14:
            onFilter = np.array(
                [
                    0.077852054085,
                    0.396539319482,
                    0.729132090846,
                    0.469782287405,
                    -0.143906003929,
                    -0.224036184994,
                    0.071309219267,
                    0.080612609151,
                    -0.038029936935,
                    -0.016574541631,
                    0.012550998556,
                    0.000429577973,
                    -0.001801640704,
                    0.000353713800,
                ]
            )
        elif Par == 16:
            onFilter = np.array(
                [
                    0.054415842243,
                    0.312871590914,
                    0.675630736297,
                    0.585354683654,
                    -0.015829105256,
                    -0.284015542962,
                    0.000472484574,
                    0.128747426620,
                    -0.017369301002,
                    -0.044088253931,
                    0.013981027917,
                    0.008746094047,
                    -0.004870352993,
                    -0.000391740373,
                    0.000675449406,
                    -0.000117476784,
                ]
            )
        elif Par == 18:
            onFilter = np.array(
                [
                    0.038077947364,
                    0.243834674613,
                    0.604823123690,
                    0.657288078051,
                    0.133197385825,
                    -0.293273783279,
                    -0.096840783223,
                    0.148540749338,
                    0.030725681479,
                    -0.067632829061,
                    0.000250947115,
                    0.022361662124,
                    -0.004723204758,
                    -0.004281503682,
                    0.001847646883,
                    0.000230385764,
                    -0.000251963189,
                    0.000039347320,
                ]
            )
        elif Par == 20:
            onFilter = np.array(
                [
                    0.026670057901,
                    0.188176800078,
                    0.527201188932,
                    0.688459039454,
                    0.281172343661,
                    -0.249846424327,
                    -0.195946274377,
                    0.127369340336,
                    0.093057364604,
                    -0.071394147166,
                    -0.029457536822,
                    0.033212674059,
                    0.003606553567,
                    -0.010733175483,
                    0.001395351747,
                    0.001992405295,
                    -0.000685856695,
                    -0.000116466855,
                    0.000093588670,
                    -0.000013264203,
                ]
            )
    if Type == "Symmlet":
        if Par == 4:
            onFilter = np.array(
                [
                    -0.107148901418,
                    -0.041910965125,
                    0.703739068656,
                    1.136658243408,
                    0.421234534204,
                    -0.140317624179,
                    -0.017824701442,
                    0.045570345896,
                ]
            )
        elif Par == 5:
            onFilter = np.array(
                [
                    0.038654795955,
                    0.041746864422,
                    -0.055344186117,
                    0.281990696854,
                    1.023052966894,
                    0.896581648380,
                    0.023478923136,
                    -0.247951362613,
                    -0.029842499869,
                    0.027632152958,
                ]
            )
        elif Par == 6:
            onFilter = np.array(
                [
                    0.021784700327,
                    0.004936612372,
                    -0.166863215412,
                    -0.068323121587,
                    0.694457972958,
                    1.113892783926,
                    0.477904371333,
                    -0.102724969862,
                    -0.029783751299,
                    0.063250562660,
                    0.002499922093,
                    -0.011031867509,
                ]
            )
        elif Par == 7:
            onFilter = np.array(
                [
                    0.003792658534,
                    -0.001481225915,
                    -0.017870431651,
                    0.043155452582,
                    0.096014767936,
                    -0.070078291222,
                    0.024665659489,
                    0.758162601964,
                    1.085782709814,
                    0.408183939725,
                    -0.198056706807,
                    -0.152463871896,
                    0.005671342686,
                    0.014521394762,
                ]
            )
        elif Par == 8:
            onFilter = np.array(
                [
                    0.002672793393,
                    -0.000428394300,
                    -0.021145686528,
                    0.005386388754,
                    0.069490465911,
                    -0.038493521263,
                    -0.073462508761,
                    0.515398670374,
                    1.099106630537,
                    0.680745347190,
                    -0.086653615406,
                    -0.202648655286,
                    0.010758611751,
                    0.044823623042,
                    -0.000766690896,
                    -0.004783458512,
                ]
            )
        elif Par == 9:
            onFilter = np.array(
                [
                    0.001512487309,
                    -0.000669141509,
                    -0.014515578553,
                    0.012528896242,
                    0.087791251554,
                    -0.025786445930,
                    -0.270893783503,
                    0.049882830959,
                    0.873048407349,
                    1.015259790832,
                    0.337658923602,
                    -0.077172161097,
                    0.000825140929,
                    0.042744433602,
                    -0.016303351226,
                    -0.018769396836,
                    0.000876502539,
                    0.001981193736,
                ]
            )
        elif Par == 10:
            onFilter = np.array(
                [
                    0.001089170447,
                    0.000135245020,
                    -0.012220642630,
                    -0.002072363923,
                    0.064950924579,
                    0.016418869426,
                    -0.225558972234,
                    -0.100240215031,
                    0.667071338154,
                    1.088251530500,
                    0.542813011213,
                    -0.050256540092,
                    -0.045240772218,
                    0.070703567550,
                    0.008152816799,
                    -0.028786231926,
                    -0.001137535314,
                    0.006495728375,
                    0.000080661204,
                    -0.000649589896,
                ]
            )
    if Type == "Vaidyanathan":
        onFilter = np.array(
            [
                -0.000062906118,
                0.000343631905,
                -0.000453956620,
                -0.000944897136,
                0.002843834547,
                0.000708137504,
                -0.008839103409,
                0.003153847056,
                0.019687215010,
                -0.014853448005,
                -0.035470398607,
                0.038742619293,
                0.055892523691,
                -0.077709750902,
                -0.083928884366,
                0.131971661417,
                0.135084227129,
                -0.194450471766,
                -0.263494802488,
                0.201612161775,
                0.635601059872,
                0.572797793211,
                0.250184129505,
                0.045799334111,
            ]
        )
    if Type == "Battle":
        if Par == 1:
            onFilterTmp = np.array(
                [
                    0.578163,
                    0.280931,
                    -0.0488618,
                    -0.0367309,
                    0.012003,
                    0.00706442,
                    -0.00274588,
                    -0.00155701,
                    0.000652922,
                    0.000361781,
                    -0.000158601,
                    -0.0000867523,
                ]
            )
        elif Par == 3:
            onFilterTmp = np.array(
                [
                    0.541736,
                    0.30683,
                    -0.035498,
                    -0.0778079,
                    0.0226846,
                    0.0297468,
                    -0.0121455,
                    -0.0127154,
                    0.00614143,
                    0.00579932,
                    -0.00307863,
                    -0.00274529,
                    0.00154624,
                    0.00133086,
                    -0.000780468,
                    -0.00065562,
                    0.000395946,
                    0.000326749,
                    -0.000201818,
                    -0.000164264,
                    0.000103307,
                ]
            )
        elif Par == 5:
            onFilterTmp = np.array(
                [
                    0.528374,
                    0.312869,
                    -0.0261771,
                    -0.0914068,
                    0.0208414,
                    0.0433544,
                    -0.0148537,
                    -0.0229951,
                    0.00990635,
                    0.0128754,
                    -0.00639886,
                    -0.00746848,
                    0.00407882,
                    0.00444002,
                    -0.00258816,
                    -0.00268646,
                    0.00164132,
                    0.00164659,
                    -0.00104207,
                    -0.00101912,
                    0.000662836,
                    0.000635563,
                    -0.000422485,
                    -0.000398759,
                    0.000269842,
                    0.000251419,
                    -0.000172685,
                    -0.000159168,
                    0.000110709,
                    0.000101113,
                ]
            )
        onFilter = np.zeros(2 * onFilterTmp.size - 1)
        onFilter[onFilterTmp.size - 1 : 2 * onFilterTmp.size] = onFilterTmp
        onFilter[0 : onFilterTmp.size - 1] = onFilterTmp[onFilterTmp.size - 1 : 0 : -1]
    return onFilter / np.linalg.norm(onFilter)


"""
 Copyright (c) 1993-5. Jonathan Buckheit and David Donoho

  Part of Wavelab Version 850
  Built Tue Jan  3 13:20:40 EST 2006
  This is Copyrighted Material
  For Copying permissions see COPYING.m
  Comments? e-mail wavelab@stat.stanford.edu
"""


def dfilters(fname, type):
    """
    This is a translation of the original Matlab implementation of dfilters.m
    from the Nonsubsampled Contourlet Toolbox. The following comment is from
    the original and only applies in so far that not all of the directional
    filters are implemented in this Python version but only those which are
    needed for the shearlet toolbox.

    DFILTERS	Generate directional 2D filters

        [h0, h1] = dfilters(fname, type)

    Input:

        fname:	Filter name.  Available 'fname' are:
                'haar':	the "Haar" filters
                'vk':	McClellan transformed of the filter
                    from the VK book
                'ko':	orthogonal filter in the Kovacevics
                    paper
                'kos':	smooth 'ko' filter
                'lax':	17 x 17 by Lu, Antoniou and Xu
                'sk':	9 x 9 by Shah and Kalker
                'cd':	7 and 9 McClellan transformed by
                                Cohen and Daubechies
                'pkva':	ladder filters by Phong et al.
                'oqf_362': regular 3 x 6 filter
           'dvmlp': regular linear phase biorthogonal filter
                    with 3 dvm
                'sinc':	ideal filter (*NO perfect recontruction*)
           'dmaxflat': diamond maxflat filters obtained from a three
                        stage ladder

             type:	'd' or 'r' for decomposition or reconstruction filters

     Output:
        h0, h1:	diamond filter pair (lowpass and highpass)

     To test those filters (for the PR condition for the FIR case), verify that:
     conv2(h0, modulate2(h1, 'b')) + conv2(modulate2(h0, 'b'), h1) = 2
     (replace + with - for even size filters)

     To test for orthogonal filter
     conv2(h, reverse2(h)) + modulate2(conv2(h, reverse2(h)), 'b') = 2

     Part of the Nonsubsampled Contourlet Toolbox
     (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """
    if fname == "haar":
        if type.lower() == "d":
            h0 = np.array([1, 1]) / np.sqrt(2)
            h1 = np.array([-1, 1]) / np.sqrt(2)
        else:
            h0 = np.array([1, 1]) / np.sqrt(2)
            h1 = np.array([1, -1]) / np.sqrt(2)
    elif fname == "vk":  # in Vetterli and Kovacevic book
        if type.lower() == "d":
            h0 = np.array([1, 2, 1]) / 4
            h1 = np.array([-1, -2, 6, -2, -1]) / 4
        else:
            h0 = np.array([-1, 2, 6, 2, -1]) / 4
            h1 = np.array([-1, 2, -1]) / 4
        t = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4  # diamon kernel
        h0 = mctrans(h0, t)
        h1 = mctrans(h1, t)
    elif fname == "ko":  # orthogonal filters in Kovacevics thesis
        a0 = 2
        a1 = 0.5
        a2 = 1
        h0 = np.array(
            [
                [0, -a1, -a0 * a1, 0],
                [-a2, -a0 * a2, -a0, 1],
                [0, a0 * a1 * a2, -a1 * a2, 0],
            ]
        )
        # h1 = qmf2(h0)
        h1 = np.array(
            [
                [0, -a1 * a2, -a0 * a1 * a2, 0],
                [1, a0, -a0 * a2, a2],
                [0, -a0 * a1, a1, 0],
            ]
        )
        # normalize filter sum and norm
        norm = np.sqrt(2) / np.sum(h0)
        h0 = h0 * norm
        h1 = h1 * norm

        if type == "r":
            # reverse filters for reconstruction
            h0 = h0[::-1]
            h1 = h1[::-1]
    elif fname == "kos":  # smooth orthogonal filters in Kovacevics thesis
        a0 = -np.sqrt(3)
        a1 = -np.sqrt(3)
        a2 = 2 + np.sqrt(3)

        h0 = np.array(
            [
                [0, -a1, -a0 * a1, 0],
                [-a2, -a0 * a2, -a0, 1],
                [0, a0 * a1 * a2, -a1 * a2, 0],
            ]
        )
        # h1 = qmf2(h0)
        h1 = np.array(
            [
                [0, -a1 * a2, -a0 * a1 * a2, 0],
                [1, a0, -a0 * a2, a2],
                [0, -a0 * a1, a1, 0],
            ]
        )
        # normalize filter sum and norm
        norm = np.sqrt(2) / np.sum(h0)
        h0 = h0 * norm
        h1 = h1 * norm

        if type == "r":
            # reverse filters for reconstruction
            h0 = h0[::-1]
            h1 = h1[::-1]
    elif fname == "lax":  # by lu, antoniou and xu
        h = np.array(
            [
                [
                    -1.2972901e-5,
                    1.2316237e-4,
                    -7.5212207e-5,
                    6.3686104e-5,
                    9.4800610e-5,
                    -7.5862919e-5,
                    2.9586164e-4,
                    -1.8430337e-4,
                ],
                [
                    1.2355540e-4,
                    -1.2780882e-4,
                    -1.9663685e-5,
                    -4.5956538e-5,
                    -6.5195193e-4,
                    -2.4722942e-4,
                    -2.1538331e-5,
                    -7.0882131e-4,
                ],
                [
                    -7.5319075e-5,
                    -1.9350810e-5,
                    -7.1947086e-4,
                    1.2295412e-3,
                    5.7411214e-4,
                    4.4705422e-4,
                    1.9623554e-3,
                    3.3596717e-4,
                ],
                [
                    6.3400249e-5,
                    -2.4947178e-4,
                    4.4905711e-4,
                    -4.1053629e-3,
                    -2.8588307e-3,
                    4.3782726e-3,
                    -3.1690509e-3,
                    -3.4371484e-3,
                ],
                [
                    9.6404973e-5,
                    -4.6116254e-5,
                    1.2371871e-3,
                    -1.1675575e-2,
                    1.6173911e-2,
                    -4.1197559e-3,
                    4.4911165e-3,
                    1.1635130e-2,
                ],
                [
                    -7.6955555e-5,
                    -6.5618379e-4,
                    5.7752252e-4,
                    1.6211426e-2,
                    2.1310378e-2,
                    -2.8712621e-3,
                    -4.8422645e-2,
                    -5.9246338e-3,
                ],
                [
                    2.9802986e-4,
                    -2.1365364e-5,
                    1.9701350e-3,
                    4.5047673e-3,
                    -4.8489158e-2,
                    -3.1809526e-3,
                    -2.9406153e-2,
                    1.8993868e-1,
                ],
                [
                    -1.8556637e-4,
                    -7.1279432e-4,
                    3.3839195e-4,
                    1.1662001e-2,
                    -5.9398223e-3,
                    -3.4467920e-3,
                    1.9006499e-1,
                    5.7235228e-1,
                ],
            ]
        )
        h0 = np.sqrt(2) * np.append(h, h[:, -2::-1], 1)
        h0 = np.append(h0, h0[-2::-1, :], 0)
        h1 = modulate2(h0, "b")
    elif fname == "sk":  # by shah and kalker
        h = np.array(
            [
                [0.621729, 0.161889, -0.0126949, -0.00542504, 0.00124838],
                [0.161889, -0.0353769, -0.0162751, -0.00499353, 0],
                [-0.0126949, -0.0162751, 0.00749029, 0, 0],
                [-0.00542504, 0.00499353, 0, 0, 0],
                [0.00124838, 0, 0, 0, 0],
            ]
        )
        h0 = np.append(h[-1:0:-1, -1:0:-1], h[-1:0:-1, :], 1)
        h0 = np.append(h0, np.append(h[:, -1:0:-1], h, 1), 0) * np.sqrt(2)
        h1 = modulate2(h0, "b")
    elif fname == "dvmlp":
        q = np.sqrt(2)
        b = 0.02
        b1 = b * b
        h = np.array(
            [
                [b / q, 0, -2 * q * b, 0, 3 * q * b, 0, -2 * q * b, 0, b / q],
                [
                    0,
                    -1 / (16 * q),
                    0,
                    9 / (16 * q),
                    1 / q,
                    9 / (16 * q),
                    0,
                    -1 / (16 * q),
                    0,
                ],
                [b / q, 0, -2 * q * b, 0, 3 * q * b, 0, -2 * q * b, 0, b / q],
            ]
        )
        g0 = np.array(
            [
                [
                    -b1 / q,
                    0,
                    4 * b1 * q,
                    0,
                    -14 * q * b1,
                    0,
                    28 * q * b1,
                    0,
                    -35 * q * b1,
                    0,
                    28 * q * b1,
                    0,
                    -14 * q * b1,
                    0,
                    4 * b1 * q,
                    0,
                    -b1 / q,
                ],
                [
                    0,
                    b / (8 * q),
                    0,
                    -13 * b / (8 * q),
                    b / q,
                    33 * b / (8 * q),
                    -2 * q * b,
                    -21 * b / (8 * q),
                    3 * q * b,
                    -21 * b / (8 * q),
                    -2 * q * b,
                    33 * b / (8 * q),
                    b / q,
                    -13 * b / (8 * q),
                    0,
                    b / (8 * q),
                    0,
                ],
                [
                    -q * b1,
                    0,
                    -1 / (256 * q) + 8 * q * b1,
                    0,
                    9 / (128 * q) - 28 * q * b1,
                    -1 / (q * 16),
                    -63 / (256 * q) + 56 * q * b1,
                    9 / (16 * q),
                    87 / (64 * q) - 70 * q * b1,
                    9 / (16 * q),
                    -63 / (256 * q) + 56 * q * b1,
                    -1 / (q * 16),
                    9 / (128 * q) - 28 * q * b1,
                    0,
                    -1 / (256 * q) + 8 * q * b1,
                    0,
                    -q * b1,
                ],
                [
                    0,
                    b / (8 * q),
                    0,
                    -13 * b / (8 * q),
                    b / q,
                    33 * b / (8 * q),
                    -2 * q * b,
                    -21 * b / (8 * q),
                    3 * q * b,
                    -21 * b / (8 * q),
                    -2 * q * b,
                    33 * b / (8 * q),
                    b / q,
                    -13 * b / (8 * q),
                    0,
                    b / (8 * q),
                    0,
                ],
                [
                    -b1 / q,
                    0,
                    4 * b1 * q,
                    0,
                    -14 * q * b1,
                    0,
                    28 * q * b1,
                    0,
                    -35 * q * b1,
                    0,
                    28 * q * b1,
                    0,
                    -14 * q * b1,
                    0,
                    4 * b1 * q,
                    0,
                    -b1 / q,
                ],
            ]
        )
        h1 = modulate2(g0, "b")
        h0 = h
        print(h1.shape)
        print(h0.shape)
        if type == "r":
            h1 = modulate2(h, "b")
            h0 = g0
    elif fname == "cd" or fname == "7-9":  # by cohen and Daubechies
        h0 = np.array(
            [
                0.026748757411,
                -0.016864118443,
                -0.078223266529,
                0.266864118443,
                0.602949018236,
                0.266864118443,
                -0.078223266529,
                -0.016864118443,
                0.026748757411,
            ]
        )
        g0 = np.array(
            [
                -0.045635881557,
                -0.028771763114,
                0.295635881557,
                0.557543526229,
                0.295635881557,
                -0.028771763114,
                -0.045635881557,
            ]
        )
        if type == "d":
            h1 = modulate2(g0, "c")
        else:
            h1 = modulate2(h0, "c")
            h0 = g0
        # use McClellan to obtain 2D filters
        t = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4  # diamond kernel
        h0 = np.sqrt(2) * mctrans(h0, t)
        h1 = np.sqrt(2) * mctrans(h1, t)
    elif fname == "oqf_362":
        h0 = (
            np.sqrt(2)
            / 64
            * np.array(
                [
                    [np.sqrt(15), -3, 0],
                    [0, 5, np.sqrt(15)],
                    [-2 * np.sqrt(2), 30, 0],
                    [0, 30, 2 * np.sqrt(15)],
                    [np.sqrt(15), 5, 0],
                    [0, -3, -np.sqrt(15)],
                ]
            )
        )
        h1 = -modulate2(h0, "b")
        h1 = -h1[::-1]
        if type == "r":
            h0 = h0[::-1]
            h1 = -modulate2(h0, "b")
            h1 = -h1[::-1]
    elif fname == "test":
        h0 = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
        h1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    elif fname == "testDVM":
        h0 = np.array([[1, 1], [1, 1]]) / np.sqrt(2)
        h1 = np.array([[-1, 1], [1, -1]]) / np.sqrt(2)
    elif fname == "qmf":  # by Lu, antoniou and xu
        # ideal response window
        m = 2
        n = 2
        w1d = np.kaiser(4 * m + 1, 2.6)
        w = np.zeros((n + m + 1, n + m + 1))
        for n1 in np.arange(-m, m + 1):
            for n2 in np.arange(-n, n + 1):
                w[n1 + m, n2 + n] = w1d[2 * m + n1 + n2] * w1d[2 * m + n1 - n2]
        h = np.zeros((n + m + 1, n + m + 1))
        for n1 in np.arange(-m, m + 1):
            for n2 in np.arange(-n, n + 1):
                h[n1 + m, n2 + n] = (
                    0.5 * np.sinc((n1 + n2) / 2) * 0.5 * np.sinc((n1 - n2) / 2)
                )
        c = np.sum(h)
        h = np.sqrt(2) * h
        h0 = h * w
        h1 = modulate2(h0, "b")
    elif fname == "qmf2":  # by Lu, Antoniou and Xu
        # ideal response window
        h = np.array(
            [
                [-0.001104, 0.002494, -0.001744, 0.004895, -0.000048, -0.000311],
                [0.008918, -0.002844, -0.025197, -0.017135, 0.003905, -0.000081],
                [-0.007587, -0.065904, 00.100431, -0.055878, 0.007023, 0.001504],
                [0.001725, 0.184162, 0.632115, 0.099414, -0.027006, -0.001110],
                [-0.017935, -0.000491, 0.191397, -0.001787, -0.010587, 0.002060],
                [0.001353, 0.005635, -0.001231, -0.009052, -0.002668, 0.000596],
            ]
        )
        h0 = h / np.sum(h)
        h1 = modulate2(h0, "b")
    elif fname == "dmaxflat4":
        M1 = 1 / np.sqrt(2)
        M2 = np.copy(M1)
        k1 = 1 - np.sqrt(2)
        k3 = np.copy(k1)
        k2 = np.copy(M1)
        h = np.array([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3]) * M1
        h = np.append(h, h[-2::-1])
        g = (
            np.array(
                [
                    -0.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                    1 + 0.5 * k1 * k2,
                ]
            )
            * M2
        )
        g = np.append(g, h[-2::-1])
        B = dmaxflat(4, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, "b")
        if type == "r":
            h1 = modulate2(h0, "b")
            h0 = g0
    elif fname == "dmaxflat5":
        M1 = 1 / np.sqrt(2)
        M2 = M1
        k1 = 1 - np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3]) * M1
        h = np.append(h, h[-2::-1])
        g = (
            np.array(
                [
                    -0.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                    1 + 0.5 * k1 * k2,
                ]
            )
            * M2
        )
        g = np.append(g, h[-2::-1])
        B = dmaxflat(5, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, "b")
        if type == "r":
            h1 = modulate2(h0, "b")
            h0 = g0
    elif fname == "dmaxflat6":
        M1 = 1 / np.sqrt(2)
        M2 = M1
        k1 = 1 - np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3]) * M1
        h = np.append(h, h[-2::-1])
        g = (
            np.array(
                [
                    -0.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                    1 + 0.5 * k1 * k2,
                ]
            )
            * M2
        )
        g = np.append(g, h[-2::-1])
        B = dmaxflat(6, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, "b")
        if type == "r":
            h1 = modulate2(h0, "b")
            h0 = g0
    elif fname == "dmaxflat7":
        M1 = 1 / np.sqrt(2)
        M2 = M1
        k1 = 1 - np.sqrt(2)
        k3 = k1
        k2 = M1
        h = np.array([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3]) * M1
        h = np.append(h, h[-2::-1])
        g = (
            np.array(
                [
                    -0.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    -0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3,
                    1 + 0.5 * k1 * k2,
                ]
            )
            * M2
        )
        g = np.append(g, h[-2::-1])
        B = dmaxflat(7, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = np.sqrt(2) * h0 / np.sum(h0)
        g0 = np.sqrt(2) * g0 / np.sum(g0)

        h1 = modulate2(g0, "b")
        if type == "r":
            h1 = modulate2(h0, "b")
            h0 = g0
    # The original file supports a case "otherwise" for unrecognized filters
    # and computes simple 1D wavelet filters for them using wfilters.m
    # I think we don't need this and skip this for the time being.
    # IN ORIGINAL MATLAB VERSION:
    # otherwise
    # % Assume the "degenerated" case: 1D wavelet filters
    # [h0,h1] = wfilters(fname, type);
    return h0 / np.sqrt(2), h1 / np.sqrt(2)


def dmaxflat(N, d):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF dmaxflat.m
    FROM THE Nonsubsampled Contourlet Toolbox.   -- Stefan Loock, Dec 2016.

    returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient, being 1 or 0
    depending on use.
    by Arthur L. da Cunha, University of Illinois Urbana-Champaign
    Aug 2004
    """
    if (N > 7) or (N < 1):
        print("Error: N must be in {1,2,...,7}")
        return 0
    if N == 1:
        h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
        h[1, 1] = d
    elif N == 2:
        h = np.array([[0, -1, 0], [-1, 0, 10], [0, 10, 0]])
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0) / 32
        h[2, 2] = d
    elif N == 3:
        h = np.array([[0, 3, 0, 2], [3, 0, -27, 0], [0, -27, 0, 174], [2, 0, 174, 0]])
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[3, 3] = d
    elif N == 4:
        h = np.array(
            [
                [0, -5, 0, -3, 0],
                [-5, 0, 52, 0, 34],
                [0, 52, 0, -276, 0],
                [-3, 0, -276, 0, 1454],
                [0, 34, 0, 1454, 0],
            ]
        ) / np.power(2, 12)
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[4, 4] = d
    elif N == 5:
        h = np.array(
            [
                [0, 35, 0, 20, 0, 18],
                [35, 0, -425, 0, -250, 0],
                [0, -425, 0, 2500, 0, 1610],
                [20, 0, 2500, 0, -10200, 0],
                [0, -250, 0, -10200, 0, 47780],
                [18, 0, 1610, 0, 47780, 0],
            ]
        ) / np.power(2, 17)
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[5, 5] = d
    elif N == 6:
        h = np.array(
            [
                [0, -63, 0, -35, 0, -30, 0],
                [-63, 0, 882, 0, 495, 0, 444],
                [0, 882, 0, -5910, 0, -3420, 0],
                [-35, 0, -5910, 0, 25875, 0, 16460],
                [0, 495, 0, 25875, 0, -89730, 0],
                [-30, 0, -3420, 0, -89730, 0, 389112],
                [0, 44, 0, 16460, 0, 389112, 0],
            ]
        ) / np.power(2, 20)
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[6, 6] = d
    elif N == 7:
        h = np.array(
            [
                [0, 231, 0, 126, 0, 105, 0, 100],
                [231, 0, -3675, 0, -2009, 0, -1715, 0],
                [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                [126, 0, 27930, 0, -136514, 0, -77910, 0],
                [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                [0, -1715, 0, -77910, 0, -1534709, 0, 6305740],
                [100, 0, 13804, 0, 311780, 0, 6305740, 0],
            ]
        ) / np.power(2, 24)
        h = np.append(h, np.fliplr(h[:, 0:-1]), 1)
        h = np.append(h, np.flipud(h[0:-1, :]), 0)
        h[7, 7] = d
    return h


def mctrans(b, t):
    """
    This is a translation of the original Matlab implementation of mctrans.m
    from the Nonsubsampled Contourlet Toolbox by Arthur L. da Cunha.

    MCTRANS McClellan transformation

        H = mctrans(B,T)

    produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.


    Convert the 1-D filter b to SUM_n a(n) cos(wn) form

    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    # if mod(n,2) != 0 -> error
    n = (b.size - 1) // 2

    b = fftlib.fftshift(b[::-1])  # inverse fftshift
    b = b[::-1]
    a = np.zeros(n + 1)
    a[0] = b[0]
    a[1 : n + 1] = 2 * b[1 : n + 1]

    inset = np.floor((np.asarray(t.shape) - 1) / 2)
    inset = inset.astype(int)
    # Use Chebyshev polynomials to compute h
    P0 = 1
    P1 = t
    h = a[1] * P1
    rows = int(inset[0] + 1)
    cols = int(inset[1] + 1)
    h[rows - 1, cols - 1] = h[rows - 1, cols - 1] + a[0] * P0
    for i in range(3, n + 2):
        P2 = 2 * signal.convolve2d(t, P1)
        rows = (rows + inset[0]).astype(int)
        cols = (cols + inset[1]).astype(int)
        if i == 3:
            P2[rows - 1, cols - 1] = P2[rows - 1, cols - 1] - P0
        else:
            P2[rows[0] - 1 : rows[-1], cols[0] - 1 : cols[-1]] = (
                P2[rows[0] - 1 : rows[-1], cols[0] - 1 : cols[-1]] - P0
            )
        rows = inset[0] + np.arange(np.asarray(P1.shape)[0]) + 1
        rows = rows.astype(int)
        cols = inset[1] + np.arange(np.asarray(P1.shape)[1]) + 1
        cols = cols.astype(int)
        hh = h
        h = a[i - 1] * P2
        h[rows[0] - 1 : rows[-1], cols[0] - 1 : cols[-1]] = (
            h[rows[0] - 1 : rows[-1], cols[0] - 1 : cols[-1]] + hh
        )
        P0 = P1
        P1 = P2
    h = np.rot90(h, 2)
    return h


def modulate2(x, type, center=np.array([0, 0])):
    """
    THIS IS A REWRITE OF THE ORIGINAL MATLAB IMPLEMENTATION OF
    modulate2.m FROM THE Nonsubsampled Contourlet Toolbox.

    MODULATE2	2D modulation

            y = modulate2(x, type, [center])

    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.

    CENTER secify the origin of modulation as floor(size(x)/2)+1+center
    (default is [0, 0])

    Part of the Nonsubsampled Contourlet Toolbox
    (http://www.mathworks.de/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox)
    """
    size = np.asarray(x.shape)
    if x.ndim == 1:
        if np.array_equal(center, [0, 0]):
            center = 0
    origin = np.floor(size / 2) + 1 + center
    n1 = np.arange(size[0]) - origin[0] + 1
    if x.ndim == 2:
        n2 = np.arange(size[1]) - origin[1] + 1
    else:
        n2 = n1
    if type == "r":
        m1 = np.power(-1, n1)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * np.transpose(np.tile(m1, (size[1], 1)))
    elif type == "c":
        m2 = np.power(-1, n2)
        if x.ndim == 1:
            y = x * m2
        else:
            y = x * np.tile(m2, np.array([size[0], 1]))
    elif type == "b":
        m1 = np.power(-1, n1)
        m2 = np.power(-1, n2)
        m = np.outer(m1, m2)
        if x.ndim == 1:
            y = x * m1
        else:
            y = x * m
    return y


def MirrorFilt(x):
    """
    This is a translation of the original Matlab implementation of
    MirrorFilt.m from the WaveLab850 toolbox.

     MirrorFilt -- Apply (-1)^t modulation
      Usage

            h = MirrorFilt(l)

      Inputs

            l   1-d signal

      Outputs

            h   1-d signal with DC frequency content shifted
                to Nyquist frequency

      Description

            h(t) = (-1)^(t-1)  * x(t),  1 <= t <= length(x)

      See Also: DyadDownHi

    Part of  WaveLab850 (http://www-stat.stanford.edu/~wavelab/)
    """
    return np.power(-1, np.arange(x.size)) * x

    """
    Copyright (c) 1993. Iain M. Johnstone

    Part of Wavelab Version 850
    Built Tue Jan  3 13:20:40 EST 2006
    This is Copyrighted Material
    For Copying permissions see COPYING.m
    Comments? e-mail wavelab@stat.stanford.edu
    """
