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
#include "setVTh_core.h"

/*
int doRunMgmt(runState *s) {
    if(s->time_step%100 == 0){
        return 1;
    }
    return 0;
}
*/
int doRunMgmt(runState *s) {
    if(s->time_step%500 == 0){
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

    // Get NeuroCore pointer
    CoreId core;
    core.id = 4;
    NeuronCore *nc = NEURON_PTR(core);

    // Read new vTh value from channel
    int vTh;
    readChannel(vThChannelID, &vTh, 1);

    // Read current vthProfileCfg entry from neuro core, change vth field and
    // rewrite vThProfileCfg field
    VthProfileStaticCfg tmpCfg = nc->vth_profile_cfg[0].vthProfileStaticCfg;
    tmpCfg.Vth = vTh;
    nc->vth_profile_cfg[0].vthProfileStaticCfg = tmpCfg;
    #CxCfg testCfg = nc->cx_cfg[0].CxCfg;
    #print(testCfg)

    // Read time of modification and send back to host
    // Note: Whenever a register write is performed an arbitrary read from that
    // same neuro core must be performed in the end to avoid deadlock.
    int time = nc->time.Time;
    writeChannel(timeChannelID, &time, 1);

    printf("Set vTh=%d at t=%d\n", vTh, time);
}
