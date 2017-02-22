# *k*-NN text classifier in Go

*k*-nearest neighbors text classifier using cosine similarity
as distance measure between documents.

## Usage

```golang
package main

import (
	"fmt"
	"github.com/akreal/knn"
)

func main() {
	knn := knn.NewKNN()

	knn.Train("Hello world!", "class1")
	knn.Train("Hello, hello.", "class2")

	k := 1

	predictedClass := knn.Predict("Say hello!", k)

	fmt.Println(predictedClass)
}
```
