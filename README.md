# SD-STGCN
spatial temporal graph convolutional network for source detection on networks

Python 3.7.5. 

tensorflow-gpu==2.3.0 for SD-STGCN

torch==1.6.0 for baselines

requirements.txt -- packages required (create a python virtual environment, activate it and pip install -r requirements.txt) 


## standard S(E)IR experiments 


Note: the following steps use ER random graph as an example. To perform experiments using other datasets described in the paper, please 
	replace ER by the proper dataset names (see the paper or ./dataset/ folder)

1. SIR simulations with length T=30 units of time on ER 1000 nodes, train and test on R0=2.5, gamma=0.4  

	1.1 data generation
	
	in ./dataset/ER/code:
		
		./run_T.sh 1000 SIR 2.5 0.4 0 30 
		
	Note: see run_T.sh for argument info 

	1.2 run model 
	
		./submit_SIR_T.sh 0 
		
	Note: the 1st argument 0 indicating graph id 0, see submit_SIR_T.sh for details. 
	Note: the last line in the output is top-1 acc, mrr, hit@5, hit@10, hit@20


2. SIR simulations with arbitrary length on ER 1000 nodes, train and test on R0=2.5, gamma=0.4 

	2.1 data generation
	
	in ./dataset/ER/code:

		./run_entire.sh 1000 SIR 2.5 0.4 0

	Note: the last argument 0 indicating graph id 0, see run_entire.sh for argument info
		
	2.2 run model
	
		./submit_SIR_entire.sh 0 
	
3. SIR simulations with arbitrary length on ER 1000 nodes, train on random R0 and gamma, test on R0=2.5, gamma=0.4

	3.1 data generation
	
	in ./dataset/ER/code:
	
		./run_entire_mix_random.sh 1000 SIR 0 
		./run_entire.sh 1000 SIR 2.5 0.4 0

	3.2 run model
	
		./submit_SIR_mix_random.sh 0


4. SEIR simulations with arbitrary length on ER 1000 nodes, train on random R0, gamma and alpha, test on R0=2.5, gamma=0.4, alpha=0.5 

	4.1 data generation
	
	in ./dataset/ER/code:
	
		./run_entire_mix_random_SEIR.sh 1000 SEIR 0
		./run_entire_SEIR.sh 1000 SEIR 2.5 0.4 0.5 0

	4.2 run model
	
		./submit_SEIR_mix_random.sh 0


## delay SIR experiments 


Note: for other datasets with different settings, change the arguments below to the proper ones

5. delay SIR simulations on ER 1000 nodes, R0=2.5 (or 10), gamma=0.4

	5.1 data generation
	
	in ./dataset/delay_SIR
		
		python gen_sim_non-Markovian_SIR_R0.py 0

	Note: run this with the 1st argument - 0 ~ 4 for 5 graphs

	5.2 run model
	
		nohup ./submit_SIR_pct_slide_nonMarkovian.sh > submit_SIR_pct_slide_nonMarkovian.log &

	5.3 print results
	
		python print_mean_std_all.py 2.5 ER

	Note: replace 2.5 by 10, for R0=10

6. delay SIR simulations on empirical contact networks (bkFratB, highSchool, sfhh)

	6.1 data generation
	
	in ./dataset/delay_SIR
		
		python gen_sim_non-Markovian_SIR_R0_emp.py bkFratB

	Note: run this with the 1st argument - bkFratB, highSchool, sfhh

	6.2 run model
		
		nohup ./submit_SIR_pct_slide_nonMarkovian_emp.sh > submit_SIR_pct_slide_nonMarkovian_emp.log &


	6.3 print results
		
		python print_mean_std_all.py 2.5 bkFratB

	Note: the 2nd argument can be bkFratB, highSchool, sfhh


## real covid-19 case data

Note: for other datasets with different settings, change the arguments below to the proper ones

7. train on SIR simulations over singapore_ER, _RGG, _config, _highSchool with random R0 and gamma, test on singapore covid-19 case data

	7.1 data generation
	
	in ./dataset/singapore_ER/code

		python preprocess.py
		./run_entire_mix_random.sh 1000 SIR 0

	Note: run this with the 3rd argument - 0 ~ 4 for 5 graphs	
	Note: for highSchool, ./run_entire_mix_random.sh 774 SIR 

	7.2 run model
		
		nohup ./submit_SIR_mix_random_singapore_ER.sh > submit_SIR_mix_random_singapore_ER.log &

	7.3 print results
		
		python print_mean_std.py singapore_ER-SIR-1000-1-15-0.1-0.4.dat


8. train on SIR simulations over tianjin_ER, _RGG, _config, _highSchool with random R0 and gamma, test on tianjin covid-19 case data

	8.1 data generation
		
	in ./dataset/tianjin_ER/code

		python preprocess.py
		./run_entire_mix_random.sh 1000 SIR 0

	Note: run this with the 3rd argument - 0 ~ 4 for 5 graphs	
	Note: for highSchool, ./run_entire_mix_random.sh 774 SIR 

	8.2 run model
		
		nohup ./submit_SIR_mix_random_tianjin_ER.sh > submit_SIR_mix_random_tianjin_ER.log &

	8.3 print results
		
		python print_mean_std.py tianjin_ER-SIR-1000-1-15-0.1-0.4.dat


## baselines

Note: codes for these methods are not publicly available. So we implemented them by ourselves. 
Note: the performance of our implementations is similar to the reported in their papers.  

9. GCN (i.e. finding_zero)

	9.1 SIR simulations on ER graph 1000 nodes, train and test on R0=2.5 and gamma=0.4

		./submit_T.sh 0

	9.2 SEIR simulations on ER graph 1000 nodes, train on random R0, gamma, alpha, test on R0=2.5, gamma=0.4, alpha=0.5

		./submit_SEIR_mix_random.sh 0

	9.3 delay SIR simulations on ER 1000 nodes, R0=2.5 (or 10), gamma=0.4

		nohup ./submit_pct_slide_R0.sh > submit_pct_slide_R0.log &

	to print results:
	
		python print_mean_std_sw_all.py ER-SIR-1000-2.5-14-sw.dat 0

	9.4 delay SIR simulations on empirical contact networks (bkFratB, highSchool, sfhh)

		nohup ./submit_pct_slide_R0_emp.sh > submit_pct_slide_R0_emp.log &

	to print results:
		
		python print_mean_std_sw_all.py bkFratB-SIR-58-2.5-14-sw.dat 1


10. DMP

	10.1 run dmp with small batches 
	
		nohup ./run_dmp_small.sh > run_dmp_small.log &

	Note: it launches a series of tasks; may take some time to finish
	Note: in run_dmp_small.sh, let the for-loop run 1 5 10 first, then run 15 20 25, to leave some resources for others
	Note: in run_dmp_small.sh, set gid=?, replace ? with 0, 1, 2, 3, 4
	Note: when it is done:
			
		mkdir results/SIR-ER-1000-T30-g?
	
	replace ? with 0, 1, 2, 3, 4 
	move all the pickle files to the corresponding dir

	10.2 print results:
		
		python evaluate_mult_g.py 1

	Note: replace the 1st argument with 1, 5, 10, ... for different steps
