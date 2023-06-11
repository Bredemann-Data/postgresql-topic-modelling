# postgresql-topic-modelling
This script is part of a 1-week project that I worked on during a Data Science qualification program.
This python script accesses a postgresql database, loads information from publihed books into a pandas dataframe, perform topic modelling and uploads the assigned topics back into a postgreSQL table.
Within the database, a view is created that shows the topic of each book by joining the topic table and the orignal table.
The postgreSQL server is accessed by methods supplied by the psycopg2 library.
Topic modelling is achieved by the Latent Dirichlet Allocation learner implemented into scikit learn.
