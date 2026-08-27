[Scope]

-> Single RSU (Road Side Unit) which acts as a global Server
-> Multiple transient vehicles that come in and out of RSU Range
-> Multiple Contexts in the same region [Day, AfterNoon, Night, Rainy, Highway, City, Sunny]
-> Contexts are simulated through Phases 
  : In Code, we are defining Phases = 5
  : We are dividing classes = 10 among these phases (iid, niid)
  : We are essentially forcing context switches purely for simulation to test code
  : We are also simulating a situation where RSU can define time based contexts itself [Day, Night, Rainy]

-> What they do out of range of RSU [Not Considered]

[Optimization]

-> Accuracy of the Global Model for the Region over all Contexts
-> Reducing Forgetting of previous contexts as we move on to new contexts
-> Convergence [Faster time to reach K Acc]
-> Reduction of Stragglers // Most Interactions

[Problem]

-> We can't do traditional CL Methods on Clients as clients are transient [We might not see same vehicle]
-> Server based CL is unexplored 
-> We have to optimize for a singular interaction with the server(region) in mind
-> We have to take care of heterogeneity in telemtry data of vehicles [e.g compute power]
-> Clients don't know which context they are in
-> Server doesn't know which context it is in

[Features]

1. Context Assignment based on Data Signature (Prototype)
- What is a prototype
In a model the last linear layer outputs a 1xFeatures=characteristics vector
We average all these features (Sum Features)/Number of Features for each feature= Feature Vector / Prototype
-> Data Signature 1xF [How we are going to assign a context]

-> Client will enter the RSU, collect enough data by its standard [We don't care what it's standards are]

Each client will calculate it's Data Signature and send to Server along with params
Server -> Has a Context Bank [prototypes] 
  : I will compare with my entire context bank and compute cosine distance and Variance [is it in the same dirn]
  : If cd < threshold Assign it that context and update that prototype in the context bank
  : Update : old_proto*0.5 + new_proto*0.5
  : If Variance is > threshold, discard it [Why? -> High variance even if in same direction will be noisy] 
  : If cd > threshold of all available proto in context bank, create new context
  -> initialize the proto of the next context with the proto that created new context

2. Low Rank Adapter Bank [An Array or Lora Params]
- What is a Low Rank Adapter [Please Look it up and Learn it]
In brief, if we have a large matrix of weights (model) 100x100, we can represent this
by a smaller multiplication of matrices A.Bt A is 100xr and B is rx100 - {R here is called rank}
[model weights] I can simply multiply it with these adapters and create a specialized section of my model

What was LorA made for:
-> To personalize pretrained models for a given context quickly
-> Side Benefit: IT is fast because latency is reduced through less comm overhead

[Model] -> First each model is divided into init_base model params, init_lora [Adapter 0] params [initalized]
Client receives base and lora params trains both sends back updated base and lora params
[Context Assignment]
: If merging with existing context -> Update LoRA Params of that context Fast [update]
-> Updated Adapter = old_lora*0.1 + new_lora*0.9
: If creating new context -> Intialize a new set of LoRA Params [Adapter 1] from the client that created context
: Always regardless of context, base model params are updated slow
-> Updated Base = old_base*0.9 + new_base*0.1

Why? -> 
: Base Model has params that will generalize over all contexts
: Adapters have params that will be specific to a certain context
: If I can find out context I can create an optimal model 

3. [Not Implemented] Client Selection if max_concurrency < num_clients
: Right now We have been working with assumption if 10 vehicles, we can handle all 10
: What if we have 50 vehicles and we can handle 10
-> Option 1 {Naive}: We can queue them up assuming all 50 will be as useful as each other in all contexts
-> Better Option:
  : Calculate Utility Score based on Telemetry Data [Including Data Signature]
  : Select 10 clients with the highest Utility Score
-> How to implement better option without increasing comm overhead and wo V2V communication
  : Network Protocol 
  : I will set an internal timer to the exp(-1*utility) Higher Utility, Lower Timer
  : My timer hits zero -> I send data and broadcast merely a signal 
  : All vehicles count recved signals, if recv signals < 10 and timer hits zero, send
  : Otherwise don't participate for avg_train_time

4. [Not Implemented] Mobility Aware Contention
: Each client knows their own avg_train_time based on their standards [We don't care how or what their stds are] 
: Based on distance_from_rsu (telemetry data) we can calculate avg_upload_time + avg_download_time
: Also from telemetry I know velocity (dirn, speed) and d_from_rsu -> I can make a guess on how long I'm here = Tstay
: If Tstay > avg_upload + avg_download + avg_train -> Participate
  

5. [Not Implemented, Might Not be Needed] Some Type of Knowledge Distillation
: In a client, penalize moving too away from what the global server knows
: Why? -> {Trying to Remove Noisy Data it might have used From outside region}
          {OR inside region not relevant to any context}


Why don't you just keep data from previous updates and use them? Whether on Client or Server
-> Client is transient so I forget
-> Server violate privacy of FL Assumptions

Why are you assuming async?
-> Sync violates vehicular situation assumptions -> How...

Last Thing
[Evaluation]
We are tracking all the contexts that we discovered each phase
Phase 0: [0, 1, 2]
Phase 1: [3]
Phase 3: [4, 5] 

Adapters : [0, 1, 2, 3, 4, 5]

I will evaluate each phase individually first
I will evaluate phase 0 using base + lora0
eval phase 0 using base + lora1, base + lora2

I will average the results to get my eval on phase 1

Eval on Phase 1:
-> Using Base + lora3

Eval on Phase 3:
-> Using Base _ loar4, base + lora5

Overall Eval:
Acc0 + acc1 + acc2 / num_phases

2 Qns should be obvious:
-> Wait so you're just going to have unlimited contexts? 
Future Work: A protocol to merge similar contexts if no. of contexts > max_contexts

Phase 0: [0, 1, 2]
Phase 1: [3]
Phase 3: [4, 5] 

-> I'm in Phase 0 Context 0, 1, 2 -> LoRA 0, 1, 2 have been created
Say my first eval occurs in Phase 0
Every eval I'm evaluating all phases
In my explanation I said if eval phas1, use base + lora3

LoRA3 isn't created, in this case, I will use the latest updated adapter to eval until lora3 is created
If right before eval is called, say context 2 was assigned and lora2 was updated, I will use lora2 for all unseen phases

Base Model is not a complete model that I can eval on [It has missing params]

Model = Base Model Params + LoRA Params


We'll quickly go through a few papers just to understand:
- What is going on currently
- What are the assumptions ppl are making and what are they violating

Scope : {FedRAV scope imporant - MultiRSU}
Assumptions
Problem : Caching {Don't Care}
        : Mobility Aware {Care a little bit}
Solution
Violations

80 percent of what your presentation on Tuesday 
