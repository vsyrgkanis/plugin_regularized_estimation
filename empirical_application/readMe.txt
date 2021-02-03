To replicate Table 1 of the manuscript, please follow the steps below.

We cannot provide the data we use because MDRC requires an application process for researcher
who wish to gain access to the public use version of the Jobs First data. In these notes and associated files, we provide computer code that allows a user to replicate our results given access to these data.

Step 1. Request the data from MDRC. We obtained this file by following the
application process described at http://www.mdrc.org/available-public-use-files and clicking on Connecticut’s Jobs First Program Analysis Data.  Our response time was 2 business days. The relevant file  "ctadmrec.sas7bdat" placed into the "data" folder of this replication package. 

Step 2. Execute prepare_data.R. This step populates the "data" folder  by "mydata.csv"

Step 3. Run main.R with synthetic=FALSE. The resulting table is under results/finaltable.txt 

Synthetic results require only Step 3 above. Run main.R with synthetic=TRUE. The resulting table is under results/finaltablesynthetic.txt. It closely mimics finaltable.txt.

To replicate how synthetic data is simulated on the basis of real data, run simulate_synthetic_data.R after Steps 1 and 2 above.



 
