package knn

import (
	"fmt"
	"github.com/reiver/go-porterstemmer"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

type Document struct {
	class int
	terms map[int]float64
}

type KNN struct {
	dictionary     map[string]int
	dictionaryLock sync.RWMutex
	postings       map[int][]int
	postingsLock   sync.RWMutex
	classes        map[string]int
	classesLock    sync.RWMutex
	documents      []*Document
	documentsLock  sync.RWMutex
}

type Neighbour struct {
	document   *Document
	similarity float64
}

type Neighbours []Neighbour

func NewKNN() *KNN {
	return &KNN{dictionary: make(map[string]int), postings: make(map[int][]int), classes: make(map[string]int), documents: make([]*Document, 0)}
}

var isAlphaNum = func(c rune) bool { return !unicode.IsLetter(c) && !unicode.IsNumber(c) }

func (knn *KNN) newDocument(text string, addTerms bool) *Document {
	document := &Document{-1, make(map[int]float64)}
	terms := make(map[int]int)

	if addTerms {
		knn.dictionaryLock.Lock()
	} else {
		knn.dictionaryLock.RLock()
	}

	for _, term := range strings.FieldsFunc(text, isAlphaNum) {
		term = porterstemmer.StemString(term)
		termId, ok := knn.dictionary[term]

		if !ok {
			if addTerms {
				termId = len(knn.dictionary)
				knn.dictionary[term] = termId
			} else {
				termId = -1
			}
		}

		terms[termId]++
	}

	if addTerms {
		knn.dictionaryLock.Unlock()
	} else {
		knn.dictionaryLock.RUnlock()
	}

	// Calculate document's magnitude
	documentMagnitude := float64(0.0)

	for _, count := range terms {
		documentMagnitude += math.Pow(float64(count), 2)
	}

	documentMagnitude = math.Sqrt(documentMagnitude)

	// Put terms to the document, normalize their counts by document's magnitude
	for term, count := range terms {
		document.terms[term] = float64(count) / documentMagnitude
	}

	return document
}

func (knn *KNN) Train(text string, class string) {
	document := knn.newDocument(text, true)

	knn.classesLock.Lock()

	if classId, ok := knn.classes[class]; !ok {
		classId = len(knn.classes)
		knn.classes[class] = classId
	}

	document.class = knn.classes[class]

	knn.classesLock.Unlock()

	knn.documentsLock.Lock()

	knn.documents = append(knn.documents, document)
	documentId := len(knn.documents) - 1

	knn.documentsLock.Unlock()

	knn.postingsLock.Lock()

	for termId, _ := range document.terms {
		knn.postings[termId] = append(knn.postings[termId], documentId)
	}

	knn.postingsLock.Unlock()
}

// Following 3 functions are needed for doing Sort() on Neighbours by similarity
func (n Neighbours) Len() int           { return len(n) }
func (n Neighbours) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }
func (n Neighbours) Less(i, j int) bool { return n[i].similarity > n[j].similarity }

func (knn *KNN) Predict(text string, k int) string {
	document := knn.newDocument(text, false)

	similarities := make(map[int]float64)

	knn.postingsLock.RLock()
	knn.documentsLock.RLock()

	for termId, _ := range document.terms {
		for documentId, _ := range knn.postings[termId] {
			similarities[documentId] += knn.documents[documentId].terms[termId] * document.terms[termId]
		}
	}

	knn.postingsLock.RUnlock()
	knn.documentsLock.RUnlock()

	neighbours := make(Neighbours, 0)

	for documentId, similarity := range similarities {
		neighbours = append(neighbours, Neighbour{knn.documents[documentId], similarity})
	}

	// Sort neighbours by similarity
	sort.Sort(neighbours)

	classCount := make(map[int]int)

	// Count classes in k first (or all) neighbours
	for i := 0; i < k && i < len(neighbours); i++ {
		classCount[neighbours[i].document.class] += 1
	}

	classId := -1
	maxClassCount := 0

	// Find the most popular class
	for id, count := range classCount {
		if count > maxClassCount {
			classId = id
			maxClassCount = count
		}
	}

	class := ""

	knn.classesLock.RLock()

	// Find name of the most popular class
	for c, i := range knn.classes {
		if i == classId {
			class = c
			break
		}
	}

	knn.classesLock.RUnlock()

	return class
}

// Helper function for Document variable printing
func (d *Document) String() string {
	return fmt.Sprintf("Document(class: %v, terms: %v)", d.class, d.terms)
}
