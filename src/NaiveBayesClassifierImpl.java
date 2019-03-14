///////////////////////////////////////////////////////////////////////////////
// Project:			 Homework 5
// Main Class File:  NewsClassifier.java
// File:             NaiveBayesClassifierImpl.java
// Semester:         CS540 Section 2 Spring 2018
//
// Author:           Meiliu Wu (mwu233@wisc.edu)
// Lecturer's Name:  Chuck Dyer
///////////////////////////////////////////////////////////////////////////////

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.lang.Math;

/**
 * The implementation of a naive bayes classifier.
 * 
 * @author Meiliu Wu
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	private Instance[] m_trainingData;
	private int m_v;
	private double m_delta;
	public int m_sports_count, m_business_count;
	public int m_sports_word_count, m_business_word_count;
	private HashMap<String, Integer> m_map[] = new HashMap[2];
	
	double[] sumV = new double[2];
	
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {

		// For all the words in the documents, count the number of occurrences. Save in
		// HashMap
		// e.g.
		// m_map[0].get("catch") should return the number of "catch" es, in the
		// documents labeled sports
		// Hint: m_map[0].get("asdasd") would return null, when the word has not
		// appeared before.
		// Use m_map[0].put(word,1) to put the first count in.
		// Use m_map[0].replace(word, count+1) to update the value
		m_trainingData = trainingData;
		m_v = v;
		m_map[0] = new HashMap<>();
		m_map[1] = new HashMap<>();

		// iterate the instances (i.e., articles) in the training data
		for (int i = 0; i < m_trainingData.length; i++) {
			Instance article = m_trainingData[i];
			// if this article belongs to sports
			if (article.label.equals(Label.SPORTS)) {
				// iterate each word in this article
				for (int j = 0; j < article.words.length; j++) {
					String word = article.words[j];
					// if this word exists in the sports hashmap
					if (m_map[0].containsKey(word)) {
						m_map[0].replace(word, m_map[0].get(word).intValue() + 1);
					}
					// if this word is a new word in the sports hashmap
					else {
						m_map[0].put(word, 1);
					}
				}
			}
			// if this article belongs to business (i.e., non-sports)
			else {
				// iterate each word in this article
				for (int j = 0; j < article.words.length; j++) {
					String word = article.words[j];
					// if this word exists in the business hashmap
					if (m_map[1].containsKey(word)) {
						m_map[1].replace(word, m_map[1].get(word).intValue() + 1);
					}
					// if this word is a new word in the sports hashmap
					else {
						m_map[1].put(word, 1);
					}
				}
			}
		}

		// |V| is the size of the total vocabulary we assume we will encounter (i.e.,
		// the dictionary size) from training data and test data
		// The value |V| will be passed to the train method of your classifier as the
		// argument int v.

		// To obtain the number of word type
		Set<String> all = new HashSet<String>();
		for (int j = 0; j < m_trainingData.length; j++) {
			for (int k = 0; k < m_trainingData[j].words.length; k++) {
				all.add(m_trainingData[j].words[k]);
			}
		}

		// To obtain the total number of words for each class(label)
		sumV[0] = 0.0;
		sumV[1] = 0.0;
		for (String w : all) {
			if (m_map[0].get(w) != null) {
				sumV[0] += m_map[0].get(w);
			}
			if (m_map[1].get(w) != null) {
				sumV[1] += m_map[1].get(w);
			}
		}
	}

	/*
	 * Counts the number of documents for each label
	 */
	public void documents_per_label_count(Instance[] trainingData) {

		m_sports_count = 0;
		m_business_count = 0;

		// iterate the instances (i.e., articles) in the training data
		for (int i = 0; i < m_trainingData.length; i++) {
			Instance article = m_trainingData[i];
			// if this article belongs to sports
			if (article.label.equals(Label.SPORTS)) {
				m_sports_count++;
			}
			// if this article belongs to business (i.e., non-sports)
			else {
				m_business_count++;
			}
		}
	}

	/*
	 * Prints the number of documents for each label
	 */
	public void print_documents_per_label_count() {
		System.out.println("SPORTS=" + m_sports_count);
		System.out.println("BUSINESS=" + m_business_count);
	}

	/*
	 * Counts the total number of words for each label
	 */
	public void words_per_label_count(Instance[] trainingData) {

		m_sports_word_count = 0;
		m_business_word_count = 0;

		for (Instance i : trainingData) {
			if (i.label.equals(Label.SPORTS)) {
				m_sports_word_count += i.words.length;
			} else {
				m_business_word_count += i.words.length;
			}
		}

	}

	/*
	 * Prints out the number of words for each label
	 */
	public void print_words_per_label_count() {
		System.out.println("SPORTS=" + m_sports_word_count);
		System.out.println("BUSINESS=" + m_business_word_count);
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPORTS) or
	 * P(BUSINESS)
	 */
	@Override
	public double p_l(Label label) {

		// Calculate the probability for the label. No smoothing here.
		// Just the number of label counts divided by the number of documents.
		double ret = 0;
		
		// set up m_sports_count and m_business_count
		documents_per_label_count(m_trainingData);

		double sum = m_sports_count + m_business_count;

		if (label == Label.SPORTS) {
			ret = m_sports_count / sum;
		} else {
			ret = m_business_count / sum;
		}

		return ret;
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPORTS) or P(word|BUSINESS)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {

		// Calculate the probability with Laplace smoothing for word in class(label)
		double ret = 0;
		m_delta = 0.00001;

		if (label == Label.SPORTS) {
			// if training data has this word
			if (m_map[0].get(word) != null) {
				ret = (m_map[0].get(word).intValue() + m_delta) / (m_v * m_delta + sumV[0]);
			}
			// training data does not have this word
			else {
				ret = m_delta / (m_v * m_delta + sumV[0]);
			}
		} else {
			// if training data has this word
			if (m_map[1].get(word) != null) {
				ret = (m_map[1].get(word).intValue() + m_delta) / (m_v * m_delta + sumV[1]);
			}
			// training data does not have this word
			else {
				ret = m_delta / (m_v * m_delta + sumV[1]);
			}
		}

		return ret;
	}

	/**
	 * Classifies an array of words as either SPORTS or BUSINESS.
	 */
	@Override
	public ClassifyResult classify(String[] words) {

		// Sum up the log probabilities for each word in the input data, and the
		// probability of the label
		// Set the label to the class with larger log probability
		ClassifyResult ret = new ClassifyResult();
		ret.label = Label.SPORTS;
		ret.log_prob_sports = 0;
		ret.log_prob_business = 0;

		for (String w : words) {
			ret.log_prob_sports += Math.log(p_w_given_l(w, Label.SPORTS));
			ret.log_prob_business += Math.log(p_w_given_l(w, Label.BUSINESS));
		}

		ret.log_prob_sports += Math.log(p_l(Label.SPORTS));
		ret.log_prob_business += Math.log(p_l(Label.BUSINESS));

		if (ret.log_prob_business > ret.log_prob_sports) {
			ret.label = Label.BUSINESS;
		}

		return ret;
	}

	/*
	 * Constructs the confusion matrix
	 */
	@Override
	public ConfusionMatrix calculate_confusion_matrix(Instance[] testData) {

		// Count the true positives, true negatives, false positives, false negatives
		int TP, FP, FN, TN;
		TP = 0;
		FP = 0;
		FN = 0;
		TN = 0;

		for (Instance ins : testData) {
			// if True SPORTS
			if (ins.label == Label.SPORTS) {
				// if TP
				if (classify(ins.words).label == Label.SPORTS) {
					TP += 1;
				}
				// else FN
				else {
					FN += 1;
				}
			}
			// if True BUSINESS
			else {
				// if TN
				if (classify(ins.words).label == Label.BUSINESS) {
					TN += 1;
				}
				// else FP
				else {
					FP += 1;
				}
			}
		}

		return new ConfusionMatrix(TP, FP, FN, TN);
	}

}
