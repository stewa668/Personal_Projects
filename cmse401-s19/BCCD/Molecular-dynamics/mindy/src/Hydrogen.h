
#ifndef HYDROGEN_H
#define HYDROGEN_H

#include <vector>

// List maintaining the global atom indicies sorted by helix groups.
class HydrogenGroupID {
  public:
    int atomID;      // global atom ID
    // isGP and atomsInGroup are determined when hydrogen bonds are found.
    int isGP;		// flag determining whether this atom is a group parent
    int atomsInGroup;   // positive number means parent of group.
                        // 0 means not in group.
    int sortVal;	// sort values (for group ordering list)
    // although the Molecule object contains get_mother_atom(), we cannot
    // use it since Molecule.h, Node.h, and structure.h would have cyclical
    // include statements.
    int GPID;	// group parent ID

    HydrogenGroupID() {};
    ~HydrogenGroupID() {};

    int operator < (const HydrogenGroupID &a) const {
      int rval;
      // check for overall group ordering
      if (sortVal != a.sortVal) rval = (sortVal < a.sortVal);
      else if (isGP)	// same group.  Check for other hydrogen ordering
	{
	// case 1: both are group parents
	if (a.isGP)	rval = (atomID < a.atomID);
	// case 2: only this atom is a group parent
	else		rval = (atomID <= a.GPID);
	}
      else
	{
	// case 3: only 'a' is a group parent
	if (a.isGP)	rval = (GPID < a.atomID);
	// case 4: both are in a group
	// case 4.1: both in different groups
	else if (GPID != a.GPID)	rval = (GPID < a.GPID);
	// case 4.2: both in same group
	else rval = (atomID < a.atomID);
	}
      return(rval);
    }

    int operator == (const HydrogenGroupID &a) const {
      // only the same when both part of same group
      return (!isGP && !a.isGP && (GPID == a.GPID) );
    }
};

typedef std::vector<HydrogenGroupID> HydrogenGroup ;

#endif

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Hydrogen.h,v $
 *      $Author: charliep $        $Locker:  $             $State: Exp $
 *      $Revision: 1.1 $     $Date: 2012/06/23 15:56:20 $
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log: Hydrogen.h,v $
 * Revision 1.1  2012/06/23 15:56:20  charliep
 * Initial load of NAMD and mindy
 *
 * Revision 1.1  1999/10/07 19:33:48  justin
 * Initial revision
 *
 * Revision 1.8  1997/12/26 23:10:50  milind
 * Made namd2 to compile, link and run under linux. Merged Templates and src
 * directoriies, and removed separate definition and declaration files for
 * templates.
 *
 * Revision 1.7  1997/06/27 21:36:47  jim
 * Fixed HydrogenGroupID operator < to ensure parents are before children.
 *
 * Revision 1.6  1997/03/21 16:35:59  nealk
 * Modified sorting case when both atoms are group members.
 * Was incorrectly sorting them by their atomID.  Now sorting by GPID.
 * Also turned off debugging in Node.C.
 *
 * Revision 1.5  1997/03/20 16:58:35  nealk
 * Atoms now sorted by: (1) hydrogen groups, (2) "groups".  The groups are
 * in the order: non-special, OH (special-1), OHH (special-2).
 * All waters (OHH) are at the end of the list.
 * Before them, all hydroxyls (OH) are together.  Although I haven't been
 * told to do this, it shouldn't hurt anything.  (And it's easier than adding
 * one more "if" to the sorting.)
 *
 * Revision 1.4  1997/03/19 20:49:20  jim
 * Changes to make inliner happy.
 *
 * Revision 1.3  1997/03/19 18:48:19  nealk
 * Forgot to close the log comment.
 *
 * Revision 1.2  1997/03/19 18:47:30  nealk
 * Added log info to Hydrogen.h
 * Fixed ComputeDPMTA.C so node 0 initializes before any other nodes register
 * with the DPMTA library.
 *
 * Revision 1.1001  1997/03/19 18:10:15  nealk
 * Added sorted hydrogen group list to molecule.
 *
 ***************************************************************************/

