/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2018 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

#include <stdlib.h>
#include <string.h>
#include "setVTh_net.h"

int doRunMgmt(runState *s) {
    if(s->time_step%100 == 0){
        return 1;
    }
    return 0;
}

void runMgmt(runState *s) {
    printf("Executing management snip to change vTh\n");
    // Get channels
    int vThChannelID = getChannelID("vThChannel");
    int timeChannelID  = getChannelID("timeChannel");
    if( vThChannelID == -1 || timeChannelID == -1) {
      printf("Invalid Channel ID\n");
      return;
    }

    // get the value 
    int vTh;
    readChannel(vThChannelID,&vTh,1);
    // set the value 
    // "nxCompartment" is a reserved key word. nxCompartment[0] refers to first compartment created in Python at NxNet level.
    nxCompartment[0].Bias = vTh;
    
    // Read time of modification and send back to host
    int time = s->time_step;
    writeChannel(timeChannelID, &time, 1);

    printf("Set vTh=%d at t=%d\n", vTh, time);
}
