# postgresql-topic-modelling
This repository contains two scripts:

script:
  This script is part of a project that I worked on during a Data Science qualification program.
  This python script accesses a postgresql database, loads information from publihed books into a pandas dataframe, perform topic modelling and uploads the assigned topics back into a postgreSQL table.
  Within the database, a view is created that shows the topic of each book by joining the topic table and the orignal table.
  The postgreSQL server is accessed by methods supplied by the psycopg2 library.
  Topic modelling is achieved by the Latent Dirichlet Allocation learner implemented into scikit learn.

script_update:
  This script uses the topic modelling pipeline from the first script and automatically assigns new topics to new entries in the database.
  This workflow is fully automated in the sense that topic modelling can be done by executing this script in the terminal.
