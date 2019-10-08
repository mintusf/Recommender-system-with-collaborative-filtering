# Recommender-system-with-collaborative-filtering
The program has two main functionalities. The first one is building a regularized recommender system with collaborative filtering and learning it with a "MovieLens" dataset (https://grouplens.org/datasets/movielens/) with 1 millions movie ratings. The second features is finding recommendations for a new user. A user, after inputting some ratings to the "Movies_fornewuser.txt" file, can use provided model, which was learnt using the first feature, to find new movies recommendations.

Files:
Python scripts:
recommender_movie.py - the main script containing all functions definitions 

Files used to create a model from scratch:
ratings.dat - a file containing 1 million ratings from MovieLens dataset
movies.dat - a file containing list of all movies used in this dataset

Files used for a new user recommendations
Features.txt - a file containing features matrix of a previously learnt recommender system using the program from this repository
Mean.txt - a file containing average ratings for all movies, used for mean normalization

Movies_fornewuser - a file in which a user who wants to learn recommendations should fill his/her ratings. The ratings should be inputed in each desired line, just after a comma. Lines with unrated movies should be left untouched. In order to obtain trustable recommendations, a user should input at least 20 ratings.
Movies_ID - a file containing ID of 300 the most popular movies
