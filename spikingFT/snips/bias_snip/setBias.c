#include <stdlib.h>
#include <string.h>
#include "setBias.h"

// need to be set
#define SIM_TIME 256
#define N_SAMPLES 256

// really constant constant
#define MAX_DECAY 4095

int simTime = SIM_TIME;
int nSamples = N_SAMPLES;

int doRunMgmt(runState *s) {
    // TODO cannot include stdio.h -> won't compile:  file reading and json functions won't work
    /*
    char buffer[100];
    FILE *fp;
    fp = fopen("./network_cornerstones.txt","r");
    fgets (buffer, 100, fp);
    fclose(fp);
     */
    if(s->time_step%simTime == 0){
        return 1;
    }
    return 0;
}

void runMgmt(runState *s) {
    printf("Executing management snip to change Bias\n");
    // Get channels
    int biasChannelID = getChannelID("biasChannel");
    int timeChannelID  = getChannelID("timeChannel");
    // int simTimeChannelID  = getChannelID("simTimeChannel");
    // int nSamplesChannelID  = getChannelID("nSamplesChannel");
    //int voltageChannelID  = getChannelID("voltageChannel");
    if( biasChannelID == -1 || timeChannelID == -1) {
      printf("Invalid Channel ID\n");
      return;
    }

    // get the value
    int bias;
    readChannel(biasChannelID,&bias,1);
    // readChannel(simTimeChannelID,&simTime,1);
    // readChannel(nSamplesChannelID,&nSamples,1);

    int i;
    for (i=0;i<(nSamples*2);++i){
        nxCompartment[i].Bias = bias;
        if (bias > 0){
            nxCompartment[i].Decay_u = MAX_DECAY;
        }
    }

    // Read time of modification and send back to host
    int time = s->time_step;
    writeChannel(timeChannelID, &time, 1);
    printf("Set bias=%d at t=%d\n", bias, time);
}
