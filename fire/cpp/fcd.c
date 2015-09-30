// TODO: Why doesn't this work?
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

int nrofcandidates = 0;
double anglemin;
int minindex;
double ratio, deltax, deltay, deltalen, deltaangle, deltadiff, deltadiff2;
double thisbeta;
double maxdsq = 4*maxr*maxr;


// Calculate anglemin of first vector
anglemin = angles(0) + M_PI - alpha;


// Find minindex with binary search
int binarymin = 0, binarymax = nrofgrads, binarymid;
while(binarymax > binarymin) {
    // Calculate midpoint
    binarymid = binarymin + ( (binarymax - binarymin) / 2 );

    if(angles(binarymid) < anglemin) {
        // Range starts in right half (has not yet started at binarymid!)
        binarymin = binarymid+1;
    } else {
        // Range has already started (and maybe ended) at binarymid
        binarymax = binarymid;
    }
}
minindex = binarymin;

// For every gradient vector:
for(int i = 0; i < nrofgrads - 1; i++) {

    // Calculate new angles
    anglemin = angles(i) + M_PI - alpha;

    // Minimum angle out of range?
    if(anglemin > M_PI)
        break;

    // Find new minimum index (by sequential search as it should be very close
    // to the current minindex)
    while(minindex < nrofgrads && angles(minindex) < anglemin)
        minindex++;

    // Break if min index out of range
    if(minindex >= nrofgrads)
        break;

    // For every possible partner grad
    for(int j = minindex; j < nrofgrads; j++) {
        // Get connection vector from j to i (!)
        deltax = x(i)-x(j);
        deltay = y(i)-y(j);
        deltalen = deltay*deltay + deltax*deltax;

        // Would resulting circle candidate be too large?
        if(deltalen > maxdsq)
            continue;

        // Convert squared diameter to radius
        deltalen = sqrt(deltalen)/2.;

        // Are norms roughly the same?
        if (norms(j) > norms(i))
          ratio = norms(j) / norms(i);
        else
          ratio = norms(i) / norms(j);
        if(ratio - 1 > gamma)
            continue;

        // Angle out of allowed range?
        if(angles(j) < angles(i) + M_PI - alpha / deltalen)
            continue;
        else if(angles(j) > angles(i) + M_PI + alpha / deltalen)
            break;

        // TODO: This is ridiculously expensive
        deltaangle = atan2(deltay, deltax);

        // Connecting line (counter-)aligned?
        // Two angles a and b are:
        //   - counter-aligned if |a| + |b| = pi  and  a*b < 0
        //   -     aligned     if   a - b   = 0
        // The line from P2 to P1 (i.e. our delta) should be aligned with v1
        // (since it's pointing outwards)
        // NOTE: Pointing outwards
        //deltadiff = fabs(deltaangle - angles(i)); // < 2pi
        // NOTE: Pointing inwards
        deltadiff2 = fabs(deltaangle - angles(j));
        thisbeta = beta / deltalen;
        // Second condition is to find pairs where one angle is roughly pi and
        // the other is roughly -pi (i.e. both are close to 180°)
        if ( deltadiff2 < thisbeta || -(deltadiff2 - 2*M_PI) < thisbeta ) {
            // We found a vector pair
            //printf("%f %f vs %f %f", x(i), y(i), x(j), y(j));
            candidates(nrofcandidates, 0) = (x(i) + x(j))/2.;
            candidates(nrofcandidates, 1) = (y(i) + y(j))/2.;
            candidates(nrofcandidates, 2) = deltalen;
            candidates(nrofcandidates, 3) = deltaangle;
            nrofcandidates++;
        }
    }
}

return_val = nrofcandidates;


