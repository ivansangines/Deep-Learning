import sys

def quicksort(arr, i, j):
 if i < j:
    pos = partition(arr, i, j)
    quicksort(arr, i, pos - 1) # quicksort left list
    quicksort(arr, pos + 1, j) # quick sort right list

def partition(arr, i, j):
 pivot = arr[j]
 small = i - 1
 for k in range(i, j):
     if arr[k] <= pivot:
        small += 1
        swap(arr, k, small)
        #printArray(arr)
 swap(arr, j, small + 1)
 print("Pivot = " + str(arr[small + 1]))
 printArray(arr)
 return small + 1

def swap(arr, i, j):
 arr[i], arr[j] = arr[j], arr[i] # exchange data at index i and j

def printArray(arr):
 print(' '.join(str(i) for i in arr))

#--------------------------------------------------------
def main():
 arr = [9, 4, 8, 3, 1, 2, 5]
 printArray(arr)
 print("Starting QuickSort :")
 
 quicksort(arr, 0, len(arr) - 1)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
