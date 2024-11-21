import tqdm
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

def run_classification( X, y, n_splits=None
                      , logreg_maxiter=200 ):

  if n_splits is None and len(np.unique(y)) < 2: return {}

  classifiers = {
    "LogisticRegression": LogisticRegression( solver='newton-cg'
                                            , max_iter=logreg_maxiter )
  , "RandomForest": RandomForestClassifier(random_state=42)
  }

  kf = StratifiedKFold(n_splits=n_splits if n_splits else len(np.unique(y)), shuffle=True, random_state=42)
  idx_split = list(kf.split(X, y))
  eval = { nm: evaluate_classifier(clf, nm, X, y, list(kf.split(X, y)))
           for nm, clf in classifiers.items()
         }
  return { nm: { 'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s)
               , 'log_loss_mean': np.mean(loglosses), 'log_loss_std': np.std(loglosses) }
           for nm, (_, f1s, loglosses, _, _) in eval.items() }

def evaluate_classifier(classifier, name, X, y, idx_splits):
  all_accuracy=[]
  all_f1=[]
  all_log_loss=[]
  all_y_test = []
  all_y_pred = []
  for train_index, test_index in (pbar:=tqdm.tqdm(idx_splits, desc=name, position=1, leave=True)):
    X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
    y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
    classifier.fit(X_train, y_train)
    all_y_test.extend(y_test)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    all_y_pred.extend(y_pred)
    all_accuracy.append(accuracy_score(y_test, y_pred))
    all_f1.append(f1_score(y_test, y_pred, average='weighted'))
    all_log_loss.append(log_loss(y_test, y_prob, labels=classifier.classes_))
    pbar.update()
  rpt = classification_report(all_y_test, all_y_pred)
  cm = confusion_matrix(all_y_test, all_y_pred)
  return all_accuracy, all_f1, all_log_loss, rpt, cm

