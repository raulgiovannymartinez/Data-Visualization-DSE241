--> How to run application?

1. Make sure the data files below are available:
	./data/data.pkl
	./data/coordinates.pkl

2. Install requirements as listed below.

3. Run python file 'app.py'

4. Open browser and enter localhost address to see application: 
	http://127.0.0.1:8050/



--> Requirements

dash                      1.12.0
dash-bootstrap-components 0.11.3
dash-core-components      1.10.0
dash-html-components      1.0.3
numpy                     1.18.4
pandas                    1.0.3      
plotly                    4.7.1      
python                    3.7.9      
python-dateutil           2.8.1      


The following instructions were tested using an anaconda environment.
STEP1	Execute the following command to create new anaconda environment called "py379" with python version 3.7.9:

	conda create -n py379 python=3.7.9

STEP2	Activate the new environment:

	conda activate py379

STEP3 Execute the following command to install all required dependencies.
	
	pip install -r requirements.txt