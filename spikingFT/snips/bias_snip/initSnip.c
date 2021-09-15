//
// Created by negin on 30.07.21.
//

#include "initSnip.h"

static int channelID = -1;

void initSnip(runState *s) {
    if(channelID == -1) {
        channelID = getChannelID("nxinit");
        if(channelID == -1) {
            printf("Invalid channelID for nxinit\n");
        }
    }

    printf("InitSnip executing");
    //nxCompartment[0].SynFormat.WgtExp = 1;
    //not a member of nxCompartment but exists in nxsdk.h ... how do I access it???
}
