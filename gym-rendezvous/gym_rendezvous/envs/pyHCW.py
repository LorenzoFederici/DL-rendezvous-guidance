import numpy as np
from numpy import sin, cos

def propagate_HCW(r0, v0, dt, omega):
    """
    This function computes the relative state vector (r, v) from the
    initial relative state vector (r0, v0) and the elapsed time dt, by using
    the closed-form solutions of the Hill-Clohessy-Wiltshire (HCW) equations.
    omega is the mean motion (angular velocity) of the target body in circular orbit.

    input:
    omega - (float) mean motion of the target (rad / s)
    r0 - (np.array) initial chaser relative position vector (km)
    v0 - (np.array) initial chaser relative velocity vector (km / s)
    dt - (float) elapsed time (s)

    output:
    r - (np.array) final chaser relative position vector (km)
    v - (np.array) final chaser relative velocity vector (km / s)
    ------------------------------------------------------------
    """

    # Components of r0 and v0 :
    x0 = r0[0]
    y0 = r0[1]
    z0 = r0[2]

    xdot0 = v0[0]
    ydot0 = v0[1]
    zdot0 = v0[2]

    # Final position vector
    x = (4. - 3.*cos(omega*dt))*x0 + (sin(omega*dt)/omega)*xdot0 + 2.*((1. - cos(omega*dt))/omega)*ydot0
    y = 6.*(sin(omega*dt) - omega*dt)*x0 + y0 - 2.*((1. - cos(omega*dt))/omega)*xdot0 + (4.*sin(omega*dt)/omega - 3.*dt)*ydot0
    z = cos(omega*dt)*z0 + (sin(omega*dt)/omega)*zdot0
    r = np.array([x, y, z])

    # Final velocity vector
    xdot = 3.*omega*sin(omega*dt)*x0 + cos(omega*dt)*xdot0 + 2.*sin(omega*dt)*ydot0
    ydot = -6*omega*(1. - cos(omega*dt))*x0 - 2.*sin(omega*dt)*xdot0 + (4.*cos(omega*dt) - 3.)*ydot0
    zdot = -omega*sin(omega*dt)*z0 + cos(omega*dt)*zdot0
    v = np.array([xdot, ydot, zdot])

    return r, v


def par2ic(coe, mu = 1.):
    """
    Description:
    
    Trasform the classic orbital element set into position / velocity vector
    Evaluate in the ECI frame


    Date:                  20 / 09 / 2013
        Revision : 2
        FIXED r_magnitude = r(not p!)
        FIXED v_z sign(now is correct!)
        Tested by : ----------


    Usage : [r, v] = coe2rvECI(COE, mu)

    Inputs :
                COE : [6]   Classical orbital elements
    [a; e; ainc; gom; pom; anu];
                mu:[1]  grav.parameter of the primary body

        ***       "COE(1)" and "mu" units must be COERENT  ***

    Outputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI

    """

    a = coe[0]
    e = coe[1]
    ainc = coe[2]
    gom = coe[3]
    pom = coe[4]
    anu = coe[5]

    p = a*(1. - pow(e,2))
    u = pom + anu
    r_norm = p / (1 + e*cos(anu))

    r[0] = r_norm*(cos(gom)*cos(u) - sin(gom)*cos(ainc)*sin(u))
    r[1] = r_norm*(sin(gom)*cos(u) + cos(gom)*cos(ainc)*sin(u))
    r[2] = r_norm*(sin(ainc)*sin(u))

    v[0] = -sqrt(mu / p)* (cos(gom)*(sin(u) + e*sin(pom)) + sin(gom)*cos(ainc)* (cos(u) + e*cos(pom)))
    v[1] = -sqrt(mu / p)* (sin(gom)*(sin(u) + e*sin(pom)) - cos(gom)*cos(ainc)* (cos(u) + e*cos(pom)))
    v[2] = -sqrt(mu / p)* (-sin(ainc)*(cos(u) + e*cos(pom)))  

    return r, v
    

def ic2par(r, v, mu = 1.0):
    """
    Description:

    Trasform the classic orbital element set into position / velocity vector
    Evaluate in the ECI frame


    Date:                  20 / 09 / 2013
        Revision : 1
        Tested by : ----------


    Usage : COE = rvECI2coe(r, v, mu)

    Inputs :
                r : [3x1]    S / C position in ECI
                v : [3x1]    S / C velocity in ECI
                mu : [1]      grav.parameter of the primary body

        ***  "r", "v", and "mu" units must be COERENT  ***

    Outputs :
                COE : [6]      Classical orbital elements
    [a; e; ainc; gom; pom; anu];
    """

    vect_rDUM = np.array(r)
    vect_vDUM = np.array(v)

    #Non dimensional units
    rconv = norm(vect_rDUM)
    vconv = sqrt(mu / rconv)

    #working with non - dimensional variables
    vect_r = vect_rDUM / rconv
    vect_v = vect_vDUM / vconv

    r_norm = norm(vect_r)
    vers_i = vect_r / r_norm
    vect_h = cross(vect_r, vect_v)

    vect_e = cross(vect_v, vect_h) - vers_i
    vect_n = cross(np.array([0, 0, 1]), vect_h)
    an = norm(vect_n)

    #_______SEMI-MAJOR AXIS_______
    av2 = dot(vect_v, vect_v)
    a = -0.5 / (av2*0.5 - 1 / r_norm)

    #_______ECCENTRICITY_______
    acca2 = dot(vect_h, vect_h)
    e2 = 1 - acca2 / a

    if (e2<-1e-8):
        e = -1
    elif (e2<1e-8):
        e = 0
    else:
        e = sqrt(e2)

    #_____ORBITAL INCLINATION______
    ainc = arccos(vect_h[2] / sqrt(acca2))

    vers_h = vect_h / sqrt(acca2)
    if (ainc < 1e-6):
        vers_n = np.array([1, 0, 0])
    else:
        vers_n = vect_n / an

    if (e2 < 1e-6):
        vers_e = vers_n
    else:
        vers_e = vect_e / e

    vers_p = cross(vers_h, vers_e)

    #_____RIGHT ASCENSION OF THE ASCENDING NODE______
    gom = arctan2(vers_n[1], vers_n[0])

    #_____ARGUMENT OF PERIAPSIS______
    pom = arctan2(dot(-vers_p,vers_n) , dot(vers_e,vers_n))

    #_____TRUE ANOMALY______
    anu = arctan2(dot(vers_i,vers_p) , dot(vers_i,vers_e))

    #Restore dimensional units
    coe = [a*rconv, e, ainc, gom, pom, anu]

    return coe