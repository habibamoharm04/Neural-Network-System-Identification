clc;
close all;
clear all;

%% Variables
n=10000;
Mu=0.1;
uk1=0;
ek1=0;
ek2=0;
ypk1=0;
ypk2=0;
ypk3=0;
ynn=0;

%% Plant Parameters
a1=1;
a2=0.1;
a3=1;
a4=0.4;
a5=1;

%% Preparing the weight function
iniwh1=-0.5+rand(1,2);
iniwh2=-0.5+rand(1,2);
iniwh3=-0.5+rand(1,2);
iniwo=-0.5+rand(1,3);
inibh=-0.5+rand(1,3);
inib01=-0.5+rand(1);

%% Assign weights and bias to the variables between -0.5 to 0.5
wh1=iniwh1;
wh2=iniwh2;
wh3=iniwh3;
wo=iniwo;
BH=inibh;
BO1=inib01;

%% Starting the learning loop
for k=1:n
    wh11=wh1(1);
    wh21=wh1(2);
    wh12=wh2(1);
    wh22=wh2(2);
    wh13=wh3(1);
    wh23=wh3(2);
    bh1=BH(1);
    bh2=BH(2);
    bh3=BH(3);
    wo11=wo(1);
    wo21=wo(2);
    wo31=wo(3);
    bo1=BO1;
    
    %% System plant
    uk=0.1*sin(2*pi*k/1000);
    yp=(a1*ypk1/(1+ypk1^2+ypk2^2))+(a2/(1+exp(a5*(-ypk1-ypk2))))+a3*uk+a4*uk1;

    %% Calculating the error
    e=yp-ynn;
    de=e-ek1;

    %% Forward direction for 2-3-1 NN identifier
    %Input layer
    x1=uk;
    x2=yp;
    Xi=[x1 x2]';
    %Hidden layers
    %at j=1
    Net1=wh1*Xi+bh1;
    OPTj1=tanh(Net1);
    %at j=2
    Net2=wh2*Xi+bh2;
    OPTj2=tanh(Net2);
    %at j=3
    Net3=wh3*Xi+bh3;
    OPTj3=tanh(Net3);
    %Output layer
    NetOPT=wo11*OPTj1+wo21*OPTj2+wo31*OPTj3+bo1;
    ynn=NetOPT;

    %% Backpropagation
    %Output layer
    dj_e=e;
    dynn_NetOPT=1;

    dNetOPT_wo11=OPTj1;
    dJ_wo11=dj_e*-dynn_NetOPT*dNetOPT_wo11;
    deltawo11=-Mu*dJ_wo11;
    wo11=wo11+deltawo11;

    dNetOPT_wo21=OPTj2;
    dJ_wo21=dj_e*-dynn_NetOPT*dNetOPT_wo21;
    deltawo21=-Mu*dJ_wo21;
    wo21=wo21+deltawo21;
    
    dNetOPT_wo31=OPTj3;
    dJ_wo31=dj_e*-dynn_NetOPT*dNetOPT_wo31;
    deltawo31=-Mu*dJ_wo31;
    wo31=wo31+deltawo31;

    wo=[wo11 wo21 wo31];

    dNetOPT_bo1=1;
    dJ_bo1=dj_e*-dynn_NetOPT*dNetOPT_bo1;
    deltabo1=-Mu*dJ_bo1;
    bo1=bo1+deltabo1;
    BO1=bo1;

    %Hidden layer
    %at j=1
    %updateWH11
    dNet_OPTj1=wo11;
    dOPTj1_Net1=1-(tanh(Net1))^2;
    dNet_wh11=x1;
    dJ_wh11=dj_e*-dynn_NetOPT*dNet_OPTj1*dOPTj1_Net1*dNet_wh11;
    deltawh11=-Mu*dJ_wh11;
    wh11=wh11+deltawh11;
    %updateWH21
    dNet_wh21=x2;
    dJ_wh21=dj_e*-dynn_NetOPT*dNet_OPTj1*dOPTj1_Net1*dNet_wh21;
    deltawh21=-Mu*dJ_wh21;
    wh21=wh21+deltawh21;
    wh1=[wh11 wh21];
    %updateBH1
    dNet_bh1=1;
    dJ_bh1=dj_e*-dynn_NetOPT*dNet_OPTj1*dOPTj1_Net1*dNet_bh1;
    deltabh1=-Mu*dJ_bh1;
    bh1=bh1+deltabh1;
    %at j=2
    %updateWH12
    dNetOPT_OPTj2=wo21;
    dOPTj2_Net2=1-(tanh(Net2))^2;
    dNet2_wh12=x1;
    dJ_wh12=dj_e*-dynn_NetOPT*dNetOPT_OPTj2*dOPTj2_Net2*dNet2_wh12;
    deltawh12=-Mu*dJ_wh12;
    wh12=wh12+deltawh12;
    %updateWH22
    dNet2_wh22=x2;
    dJ_wh22=dj_e*-dynn_NetOPT*dNetOPT_OPTj2*dOPTj2_Net2*dNet2_wh22;
    deltawh22=-Mu*dJ_wh22;
    wh22=wh22+deltawh22;
    wh2=[wh12 wh22];
    %updateBH2
    dNet2_bh2=1;
    dJ_bh2=dj_e*-dynn_NetOPT*dNetOPT_OPTj2*dOPTj2_Net2*dNet2_bh2;
    deltabh2=-Mu*dJ_bh2;
    bh2=bh2+deltabh2;
    %at j=3
    %updateWH13
    dNetOPT_OPTj3=wo31;
    dOPTj3_Net3=1-(tanh(Net3))^2;
    dNet3_wh13=x1;
    dJ_wh13=dj_e*-dynn_NetOPT*dNetOPT_OPTj3*dOPTj3_Net3*dNet3_wh13;
    deltawh13=-Mu*dJ_wh13;
    wh13=wh13+deltawh13;
    %updateWH23
    dNet3_wh23=x2;
    dJ_wh23=dj_e*-dynn_NetOPT*dNetOPT_OPTj3*dOPTj3_Net3*dNet3_wh23;
    deltawh23=-Mu*dJ_wh23;
    wh23=wh23+deltawh23;
    wh3=[wh13 wh23];
    %updateBH3
    dNet3_bh3=1;
    dJ_bh3=dj_e*-dynn_NetOPT*dNetOPT_OPTj3*dOPTj3_Net3*dNet3_bh3;
    deltabh3=-Mu*dJ_bh3;
    bh3=bh3+deltabh3;
    BH=[bh1 bh2 bh3];

    %% Updating the last variable value
    uk1=uk;
    ypk3=ypk2;
    ypk2=ypk1;
    ypk1=yp;
    ek1=e;

    %% Storing the results
    ynnk1=ynn;
    YP(k)=yp;
    YNN(k)=ynn;

    %% Print the values for each iteration
    NETWORK=[yp ynn];
    disp(NETWORK);
end