// Two independent tabular Q learners; accelerated simulation only, no neural network.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using Table = std::array<std::array<double,15>,225>;
int greedy(const std::array<double,15>& row){return int(std::max_element(row.begin(),row.end())-row.begin());}
int main(int argc,char**argv){
 if(argc!=11){std::cerr<<"sessions delta alpha beta max_steps stable_steps seed pN pM output_prefix\n";return 2;}
 int sessions=std::stoi(argv[1]),cap=std::stoi(argv[5]),stable=std::stoi(argv[6]);
 double delta=std::stod(argv[2]),alpha=std::stod(argv[3]),beta=std::stod(argv[4]);
 uint64_t seed=std::stoull(argv[7]);double pn=std::stod(argv[8]),pm=std::stod(argv[9]);std::string prefix=argv[10];
 std::array<double,15> price;double pay[2][15][15];
 for(int a=0;a<15;++a)price[a]=pn-.1*(pm-pn)+a*(1.2*(pm-pn))/14.;
 for(int a=0;a<15;++a)for(int b=0;b<15;++b){double u=std::exp((2-price[a])/.25),v=std::exp((2-price[b])/.25),den=1+u+v;pay[0][a][b]=(price[a]-1)*u/den;pay[1][a][b]=(price[b]-1)*v/den;}
 std::ofstream summary(prefix+"_sessions.csv"),trace(prefix+"_learning.csv");summary<<"session,seed,steps,converged,last_state\n";trace<<"session,step,price,profit\n";trace<<std::setprecision(16);
 for(int session=0;session<sessions;++session){
  std::mt19937_64 rng(seed+session);std::uniform_real_distribution<double> unif(0.,1.);std::uniform_int_distribution<int> action(0,14),state_dist(0,224);
  Table q[2];std::array<int,225> policy[2];
  for(int i=0;i<2;++i)for(int s=0;s<225;++s){for(int a=0;a<15;++a){double sum=0;for(int b=0;b<15;++b)sum+=(i==0?pay[0][a][b]:pay[1][b][a]);q[i][s][a]=sum/(15*(1-delta));}policy[i][s]=greedy(q[i][s]);}
  int s=(seed==0?7:state_dist(rng)),unchanged=0,steps=0;double sum_p=0,sum_profit=0;int window=0;
  for(int t=0;t<cap;++t){
   double epsilon=std::exp(-beta*t);int a,b; if(seed==0){a=t%15;b=(t*7+3)%15;}else{a=unif(rng)<epsilon?action(rng):policy[0][s];b=unif(rng)<epsilon?action(rng):policy[1][s];}int next=a*15+b;
   int actions[2]={a,b};bool change=false;
   for(int i=0;i<2;++i){double target=pay[i][a][b]+delta*q[i][next][policy[i][next]];q[i][s][actions[i]]=(1-alpha)*q[i][s][actions[i]]+alpha*target;int new_greedy=greedy(q[i][s]);if(new_greedy!=policy[i][s])change=true;policy[i][s]=new_greedy;}
   unchanged=change?0:unchanged+1;s=next;steps=t+1;
   sum_p+=(price[a]+price[b])/2;sum_profit+=(pay[0][a][b]+pay[1][a][b])/2;++window;
   if(window==5000||unchanged>=stable||t==cap-1){trace<<session<<','<<steps<<','<<sum_p/window<<','<<sum_profit/window<<'\n';sum_p=sum_profit=0;window=0;}
   if(unchanged>=stable)break;
  }
  summary<<session<<','<<(seed+session)<<','<<steps<<','<<(unchanged>=stable)<<','<<s<<'\n';
  std::ofstream file(prefix+"_q_"+std::to_string(session)+".bin",std::ios::binary);
  for(int i=0;i<2;++i)for(int st=0;st<225;++st)file.write(reinterpret_cast<const char*>(q[i][st].data()),15*sizeof(double));
 }
}
