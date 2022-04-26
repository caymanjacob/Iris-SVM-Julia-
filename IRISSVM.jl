################################################################################
# Defuse Labs "Iris SVM" - Jacob Appleton - April 18, 2022
################################################################################

# load packages by: ] -> add "x"
using LIBSVM, RDatasets
using LinearAlgebra, Random, Statistics
using DataVoyager
using DataFrames
using DelimitedFiles
using Plots
using DataStructures

# Load data by choosing dataset: 
d = dataset("datasets", "iris")

# Begining of ui
println("Welcome to the Iris SVM, here is the table of flowers that can be tested on:")
readline()
vscodedisplay(d)
println("Please choose a number between 10 and 50 for the flowers you would to leave out of each kind
")
lra = readline()
lta = parse(Int, chomp(lra))

# Pick random amount to test and show
ra = lta #rand(10:50)
x3 = (50 - lta) * 3

println("
Testing on: ", (50 - ra) * 3, " flowers") 

println("
Graph of the flowers we have to choose from

Red : Setosa
Green : Versicolor
Purple : Virginica")

# Testing set we start off with to learn from
pp = d[1:50 - ra, 1], d[1:50 - ra, 3], d[1:50 - ra, 2]
dd = d[51:100 - ra, 1], d[51:100 - ra, 3], d[51:100 - ra, 2]
yy = d[101:150 - ra, 1], d[101:150 - ra, 3], d[101:150 - ra, 2]
ox = 1, 1, 1

scatter([ ox, pp, dd, yy])

readline()

println("Scaling Data (Gausian Distribution)...")
################################################################################

# Implament a random number upon dataset
f1s = d[1:50 - ra, 1]
f1v = d[51:100 - ra, 1]
f1i = d[101:150 - ra, 1]
f2s = d[1:50 - ra, 2]
f2i = d[101:150 - ra, 2]
f2v = d[51:100 - ra, 2]
f3s = d[1:50 - ra, 3]
f3i = d[101:150 - ra, 3]
f3v = d[51:100 - ra, 3]

# Feature Scaling - standartization
f1sb = mean(f1s)
f1vb = mean(f1v)
f1ib = mean(f1i)
f2sb = mean(f2s)
f2vb = mean(f2v)
f2ib = mean(f2i)
f3sb = mean(f3s)
f3vb = mean(f3v)
f3ib = mean(f3i)

# Deviation
f1ss = std(f1s)
f1vs = std(f1v)
f1is = std(f1i)
f2ss = std(f2s)
f2vs = std(f2v)
f2is = std(f2i)
f3ss = std(f3s)
f3vs = std(f3v)
f3is = std(f3i)

# Gausian distribution 
f1sg = (f1s .- f1sb) ./ f1ss
f1vg = (f1v .- f1vb) ./ f1vs
f1ig = (f1i .- f1ib) ./ f1is
f2sg = (f2s .- f2sb) ./ f2ss
f2vg = (f2v .- f2vb) ./ f2vs
f2ig = (f2i .- f2ib) ./ f2is
f3sg = (f3s .- f3sb) ./ f3ss
f3vg = (f3v .- f3vb) ./ f3vs
f3ig = (f3i .- f3ib) ./ f3is

# Combine types
sl = string(f1sg, f1vg, f1ig, ", ")
sw = string(f2sg, f2vg, f2ig, ", ")
pl = string(f3sg, f3vg, f3ig, ", ")

# Removes junk from combination
SLl = replace(sl, "][" => ",")
SWw = replace(sw, "][" => ",")
PLl = replace(pl, "][" => ",")

SLL = replace(SLl, "]" => "")
SWW = replace(SWw, "]" => "")
PLL = replace(PLl, "]" => "")

SL = replace(SLL, "[" => "")
SW = replace(SWW, "[" => "")
PL = replace(PLL, "[" => "")

# Split srting (screw the string)
ssl = split(SL, ",")
ssw = split(SW, ",")
spl = split(PL, ",")

# Set names
Setosa = "Setosa,"
Versicolor = "Versicolor,"
Virginica = "Virginica,"

# Provide name column
hs = repeat(Setosa, 50 - ra)
hv = repeat(Versicolor, 50 - ra)
hi = repeat(Virginica, 50 - ra)

# Combine name column
Stringnames = string(hs, hv, hi)
# Split name column
ar = split(Stringnames, ",")

# New dataframe
de = DataFrame(
    Sepal_Length = ssl, 
    Sepal_Width = ssw,
    Petal_Length = spl,
    Species = ar
)

# Remove extra row at end
df = de[1:x3, :]

println("Table of reditributed data:")
readline()

vscodedisplay(df)
#Voyager(df)

# Split data?
X = Matrix(d[:, 1:3])
y = d.Species
#vscodedisplay(X)

# Ordering
sx = f1sg, f2sg, f3sg
vx = f1vg, f2vg, f3vg
ix = f1ig, f2ig, f3ig
oui = 1, 1, 1

println("Graph of the re-distributed data")

# Feature scale of the flowers we know
scatter([oui, sx, vx, ix])
readline()

################################################################################

println("Learning . . .")

# Matrix
tri = [f1sg f1vg f1ig f2sg f2vg f2ig f3sg f3vg f3ig]
#vscodedisplay(tri)

#Define function to split data 
function perclass_splits(y, percent)
    u = unique(y)
    keep_index = []
    for class in u
        i = findall(y .== class)
        r = randsubseq(i, percent)
        push!(keep_index, r...)
    end
    return keep_index
end

# Split data for learning 
Random.seed!(1)
train_index = perclass_splits(y, (ra + 50) * 0.01)
test_index = setdiff(1:length(y), train_index)

#######################################################################
#vscodedisplay(test_index)
#vscodedisplay([train_index])
#######################################################################

#vscodedisplay(Xtest)

#actual SVM
Xtrain = X[train_index, :]

Xtest = X[test_index, :]

ytrain = y[train_index]

ytest = y[test_index]

# Transpose
Xtrain_t = Xtrain'

Xtest_t = Xtest'

# Run model
m = svmtrain(Xtrain_t, ytrain)

# Make predictions
y_hat, decision_values = svmpredict(m, Xtest_t)

# Accuracy
g = mean(y_hat .== ytest)
q = g * 100

# Check display 
che = [y_hat[i] == ytest[i] for i in 1:length(y_hat)]


# Split up o.g. matrix info for new one
sepall = Xtest[:, 1]
sepalw = Xtest[:, 2]
petall = Xtest[:, 3]

# New dataframe with true or false
check_display = DataFrame(
    Prediction_of_Species= y_hat, 
    Actual_Species = ytest,
    Accuracy = che,
    Sepal_Length = sepall,
    Sepal_Width = sepalw,
    Petal_Length = petall
)
# Show new dataframe with what's false
#vscodedisplay(check_display)

# Count how many are known
kf = nrow(check_display)

yu = string(y_hat)
hat = DataFrame(y_hat)
vscodedisplay(hat)

yi = replace(yu, "\"versicolor\"" => "2")
yo = replace(yi, "\"setosa\"" => "1")
yk = replace(yo, "\"virginica\"" => "3")
yr = replace(yk, "CategoricalArrays.CategoricalValue{String, UInt8}[" => "")
ht = replace(yr, "," => "")
yh = replace(ht, "]" => "")

oy = replace(yh, "2" => "")
oyt = replace(oy, "3" => "")


# Display the divide
println(kf, " flowers known, finding ", ((50 - ra) * 3) - kf, " flowers...")

println("Done!")

# The answer
#pt = check_display[1:kf ./ 3, 4], check_display[1:50 - ra, 6], check_display[1:50 - ra, 5]
#ddt = check_display[1:50 - ra:(50 - ra) * 2, 4], check_display[1:50 - ra:(50 - ra) * 2, 6], check_display[1:50 - ra:(50 - ra) * 2, 5]
#yyt = check_display[(50 - ra) * 2:(50 - ra) * 3, 4], check_display[(50 - ra) * 2:(50 - ra) * 3, 6], check_display[(50 - ra) * 2:(50 - ra) * 3, 5]
#xxx = 1, 1, 1
#scatter([xxx, ppt, ddt, yyt])

# UI
println("The accuracy is ", round(q; digits = 2), "%")

println("Table of predictions:")
readline()
vscodedisplay(check_display)

println("Graph of predictions:")
scatter([ pp, dd, yy])

#deal
# Testing set we start off with to learn from
#pp = d[1:50 - ra, 1], d[1:50 - ra, 3], d[1:50 - ra, 2]
#dd = d[51:100 - ra, 1], d[51:100 - ra, 3], d[51:100 - ra, 2]
#yy = d[101:150 - ra, 1], d[101:150 - ra, 3], d[101:150 - ra, 2]
#ox = d[1:150, 1], d[1:150, 3], d[1:150, 2]
#scatter([ pp, dd, yy])