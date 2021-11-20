from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


models = {
	"n_bayes": OneVsRestClassifier(MultinomialNB())
}