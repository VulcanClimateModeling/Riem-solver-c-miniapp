from gt4py.gtscript import (__INLINED, BACKWARD, FORWARD, IJ, IJK, PARALLEL,
                            Field, I, J, K, computation, exp, interval, log)

MOIST_CAPPA: bool = False
USE_COND: bool = False
GRAV: float = 9.8
RDGAS: float = 287.04


def calc_gama(akap):
    return 1. / (1. - akap)


def calc_t1g(dt, gama):
    if __INLINED(MOIST_CAPPA):
        t1g = 2. * dt * dt
    else:
        t1g = gama * 2. * dt * dt
    return t1g


def calc_dz2(dm2, pt, cp2, akap, p_fac, pm2, p1):
    if __INLINED(MOIST_CAPPA):
        fac = cp2 - 1.
    else:
        fac = akap - 1
    return -dm2 * RDGAS * pt * exp(fac * log(max(p_fac * pm2, p1 + pm2)))


def riem_solver_c(
        cappa: Field[float, IJK], hs: Field[float, IJ], w3: Field[float, IJK],
        pt: Field[float, IJK], q_con: Field[float, IJK], delp: Field[float, IJK],
        gz: Field[float, IJK], pef: Field[float, IJK], ws: Field[float, IJK],
        pe: Field[float, IJK], p_fac: float, scale_m: float, ms: int, dt: float,
        akap: float, cp: float, ptop: float):
    """
    C-grid Riemann solver.

    Args:
        ms: ...
        dt: Timestep
        akap: TBD
        cappa: TBD
        cp: TBD
        ptop: TBD
        hs: TBD
        w3: TBD
        pt: TBD
        q_con: TBD
        delp: TBD
        gz: TBD [inout]
        pef: TBD [out]
        ws: TBD
        p_fac: TBD
        scale_m: TBD
    """
    from __externals__ import A_IMP

    # Line 121
    with computation(PARALLEL), interval(0, -1):
        dm = delp

    # Line 129
    with computation(FORWARD):
        with interval(0, 1):
            pef = ptop
            pem = ptop
            if __INLINED(USE_COND):
                peg = ptop

        with interval(1, None):
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            if __INLINED(USE_COND):
                peg = peg[0, 0, -1] + dm[0, 0, -1] * (1. - q_con[0, 0, -1])

    # Line 151
    with computation(PARALLEL), interval(0, -1):
        dz2 = gz[0, 0, 1] - gz[0, 0, 0]

        if __INLINED(USE_COND):
            pm2 = (peg[0, 0, 1] - peg[0, 0, 0]) / log(peg[0, 0, 1] / peg[0, 0, 0])
            if __INLINED(MOIST_CAPPA):
                cp2 = cappa
                gm2 = 1. / (1. - cp2)
        else:
            pm2 = dm / log(pem[0, 0, 1] / pem[0, 0, 0])

        dm = dm * 1. / GRAV
        w2 = w3

    # SIM1_solver {
    # Line 232
    with computation(PARALLEL), interval(0, -1):
        if __INLINED(A_IMP > 0.5):
            if __INLINED(MOIST_CAPPA):
                fac = gm2
            else:
                fac = calc_gama(akap)
            pe = exp(fac * log(-dm / dz2 * RDGAS * pt)) - pm2

            w1 = w2

    # Line 245
    with computation(PARALLEL), interval(0, -2):
        if __INLINED(A_IMP > 0.5):
            g_rat = dm[0, 0, 0] / dm[0, 0, 1]
            bb = 2. * (1. + g_rat)
            dd = 3. * (pe + g_rat) * pe[0, 0, 1]

    # Line 255
    with computation(FORWARD):
        # Set special conditions on pp, bb, dd
        with interval(0, 1):
            if __INLINED(A_IMP > 0.5):
                bet = bb
                pp = 0
        with interval(1, 2):
            if __INLINED(A_IMP > 0.5):
                pp = dd[0, 0, -1] / bb[0, 0, -1]
        with interval(-2, -1):
            if __INLINED(A_IMP > 0.5):
                bb = 2.
                dd = 3 * pe

    # Line 265
    with computation(PARALLEL), interval(1, -1):
        if __INLINED(A_IMP > 0.5):
            gam = g_rat[0, 0, -1] / bet[0, 0]
            bet = bb - gam

    with computation(PARALLEL), interval(2, None):
        if __INLINED(A_IMP > 0.5):
            pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0]

    # Line 275 (fused with 284)
    with computation(BACKWARD), interval(1, -1):  # this may need to be -2
        if __INLINED(A_IMP > 0.5):
            pp = pp - gam * pp[0, 0, 1]

            gama = calc_gama(akap)
            t1g = calc_t1g(dt, gama)

            aa = t1g * (pem[0, 0, 0] + pp[0, 0, 0]) / (dz2[0, 0, -1] + dz2[0, 0, 0])

            if __INLINED(MOIST_CAPPA):
                aa *= 0.5 * (gm2[0, 0, -1] + gm2[0, 0, 0])

    # Line 295
    with computation(PARALLEL), interval(0, 1):
        if __INLINED(A_IMP > 0.5):
            bet = dm[0, 0, 0] - aa[0, 0, 1]
            w2 = (dm[0, 0, 0] * w1[0, 0, 0] + dt * pp[0, 0, 1]) / bet

    # Line 302
    with computation(FORWARD), interval(1, -2):
        if __INLINED(A_IMP > 0.5):
            gam = aa / bet
            bet = dm - (aa * (gam + 1) + aa[0, 0, 1])
            w2 = (dm * w1 + dt * (pp[0, 0, 1] - pp[0, 0, 0]) - aa * w2[0, 0, -1]) / bet

    # Line 312
    with computation(FORWARD), interval(-2, -1):
        if __INLINED(A_IMP > 0.5):
            t1g = calc_t1g(dt, gam)
            p1 = t1g / dz2 * (pem[0, 0, 1] + pp[0, 0, 1])
            if __INLINED(MOIST_CAPPA):
                p1 *= gm2
            gam = aa / bet
            bet = dm - (aa * (1.0 + gam) + p1)
            w2 = (dm * w1 + dt * (pp[0, 0, 1] - pp[0, 0, 0]
                                  ) - p1 * ws - aa * w2[0, 0, -1]) / bet

    # Line 325
    with computation(BACKWARD), interval(0, -2):
        if __INLINED(A_IMP > 0.5):
            w2 -= gam[0, 0, 1] * w2[0, 0, 1]

    # Line 332
    with computation(FORWARD):
        with interval(0, 1):
            if __INLINED(A_IMP > 0.5):
                pe = 0.
        with interval(1, -1):
            if __INLINED(A_IMP > 0.5):
                rdt = 1. / dt
                pe = pe[0, 0, -1] + dm * (w2 - w1) * rdt

    # Line 346
    with computation(BACKWARD):
        with interval(-2, -1):
            if __INLINED(A_IMP > 0.5):
                r3 = 1. / 3.
                p1 = (pe + 2. * pe[0, 0, 1]) * r3
                dz2 = calc_dz2(dm, pt, cp2, akap, p_fac, pm2, p1)
        with interval(0, -2):
            if __INLINED(A_IMP > 0.5):
                r3 = 1. / 3.
                p1 = (pe + bb * pe[0, 0, 1] + g_rat * pe[0, 0, 2]) * r3 - g_rat * p1
                dz2 = calc_dz2(dm, pt, cp2, akap, p_fac, pm2, p1)

    # } SIM1_solver

    # Line 177
    with computation(PARALLEL), interval(1, None):
        pef = pe + pem

    with computation(BACKWARD):
        with interval(-1, None):
            gz = hs
        with interval(1, -1):
            gz = gz[0, 0, 1] - dz2 * GRAV
