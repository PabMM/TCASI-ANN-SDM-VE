ўИ#
Т
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
ћ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8щэ
z
regression/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameregression/bias/v
s
%regression/bias/v/Read/ReadVariableOpReadVariableOpregression/bias/v*
_output_shapes
:*
dtype0

regression/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameregression/kernel/v
{
'regression/kernel/v/Read/ReadVariableOpReadVariableOpregression/kernel/v*
_output_shapes

: *
dtype0

layer_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_7/beta/v

0layer_normalization_7/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta/v*
_output_shapes
: *
dtype0

layer_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_7/gamma/v

1layer_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma/v*
_output_shapes
: *
dtype0
t
dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_7/bias/v
m
"dense_7/bias/v/Read/ReadVariableOpReadVariableOpdense_7/bias/v*
_output_shapes
: *
dtype0
|
dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_7/kernel/v
u
$dense_7/kernel/v/Read/ReadVariableOpReadVariableOpdense_7/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_6/beta/v

0layer_normalization_6/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta/v*
_output_shapes
: *
dtype0

layer_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_6/gamma/v

1layer_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma/v*
_output_shapes
: *
dtype0
t
dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias/v
m
"dense_6/bias/v/Read/ReadVariableOpReadVariableOpdense_6/bias/v*
_output_shapes
: *
dtype0
|
dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_6/kernel/v
u
$dense_6/kernel/v/Read/ReadVariableOpReadVariableOpdense_6/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_5/beta/v

0layer_normalization_5/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta/v*
_output_shapes
: *
dtype0

layer_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_5/gamma/v

1layer_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma/v*
_output_shapes
: *
dtype0
t
dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias/v
m
"dense_5/bias/v/Read/ReadVariableOpReadVariableOpdense_5/bias/v*
_output_shapes
: *
dtype0
|
dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_5/kernel/v
u
$dense_5/kernel/v/Read/ReadVariableOpReadVariableOpdense_5/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_4/beta/v

0layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta/v*
_output_shapes
: *
dtype0

layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_4/gamma/v

1layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma/v*
_output_shapes
: *
dtype0
t
dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias/v
m
"dense_4/bias/v/Read/ReadVariableOpReadVariableOpdense_4/bias/v*
_output_shapes
: *
dtype0
|
dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_4/kernel/v
u
$dense_4/kernel/v/Read/ReadVariableOpReadVariableOpdense_4/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_3/beta/v

0layer_normalization_3/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta/v*
_output_shapes
: *
dtype0

layer_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_3/gamma/v

1layer_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma/v*
_output_shapes
: *
dtype0
t
dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias/v
m
"dense_3/bias/v/Read/ReadVariableOpReadVariableOpdense_3/bias/v*
_output_shapes
: *
dtype0
|
dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_3/kernel/v
u
$dense_3/kernel/v/Read/ReadVariableOpReadVariableOpdense_3/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_2/beta/v

0layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta/v*
_output_shapes
: *
dtype0

layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_2/gamma/v

1layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma/v*
_output_shapes
: *
dtype0
t
dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias/v
m
"dense_2/bias/v/Read/ReadVariableOpReadVariableOpdense_2/bias/v*
_output_shapes
: *
dtype0
|
dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namedense_2/kernel/v
u
$dense_2/kernel/v/Read/ReadVariableOpReadVariableOpdense_2/kernel/v*
_output_shapes

:@ *
dtype0

layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_1/beta/v

0layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta/v*
_output_shapes
: *
dtype0

layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_1/gamma/v

1layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma/v*
_output_shapes
: *
dtype0
t
dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias/v
m
"dense_1/bias/v/Read/ReadVariableOpReadVariableOpdense_1/bias/v*
_output_shapes
: *
dtype0
|
dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_1/kernel/v
u
$dense_1/kernel/v/Read/ReadVariableOpReadVariableOpdense_1/kernel/v*
_output_shapes

:  *
dtype0

layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization/beta/v

.layer_normalization/beta/v/Read/ReadVariableOpReadVariableOplayer_normalization/beta/v*
_output_shapes
: *
dtype0

layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization/gamma/v

/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOplayer_normalization/gamma/v*
_output_shapes
: *
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
: *
dtype0
x
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel/v
q
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes

: *
dtype0

batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta/v

.batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/v*
_output_shapes
:*
dtype0

batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma/v

/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/v*
_output_shapes
:*
dtype0
z
regression/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameregression/bias/m
s
%regression/bias/m/Read/ReadVariableOpReadVariableOpregression/bias/m*
_output_shapes
:*
dtype0

regression/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameregression/kernel/m
{
'regression/kernel/m/Read/ReadVariableOpReadVariableOpregression/kernel/m*
_output_shapes

: *
dtype0

layer_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_7/beta/m

0layer_normalization_7/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta/m*
_output_shapes
: *
dtype0

layer_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_7/gamma/m

1layer_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma/m*
_output_shapes
: *
dtype0
t
dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_7/bias/m
m
"dense_7/bias/m/Read/ReadVariableOpReadVariableOpdense_7/bias/m*
_output_shapes
: *
dtype0
|
dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_7/kernel/m
u
$dense_7/kernel/m/Read/ReadVariableOpReadVariableOpdense_7/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_6/beta/m

0layer_normalization_6/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta/m*
_output_shapes
: *
dtype0

layer_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_6/gamma/m

1layer_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma/m*
_output_shapes
: *
dtype0
t
dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias/m
m
"dense_6/bias/m/Read/ReadVariableOpReadVariableOpdense_6/bias/m*
_output_shapes
: *
dtype0
|
dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_6/kernel/m
u
$dense_6/kernel/m/Read/ReadVariableOpReadVariableOpdense_6/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_5/beta/m

0layer_normalization_5/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta/m*
_output_shapes
: *
dtype0

layer_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_5/gamma/m

1layer_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma/m*
_output_shapes
: *
dtype0
t
dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias/m
m
"dense_5/bias/m/Read/ReadVariableOpReadVariableOpdense_5/bias/m*
_output_shapes
: *
dtype0
|
dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_5/kernel/m
u
$dense_5/kernel/m/Read/ReadVariableOpReadVariableOpdense_5/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_4/beta/m

0layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta/m*
_output_shapes
: *
dtype0

layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_4/gamma/m

1layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma/m*
_output_shapes
: *
dtype0
t
dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias/m
m
"dense_4/bias/m/Read/ReadVariableOpReadVariableOpdense_4/bias/m*
_output_shapes
: *
dtype0
|
dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_4/kernel/m
u
$dense_4/kernel/m/Read/ReadVariableOpReadVariableOpdense_4/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_3/beta/m

0layer_normalization_3/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta/m*
_output_shapes
: *
dtype0

layer_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_3/gamma/m

1layer_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma/m*
_output_shapes
: *
dtype0
t
dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias/m
m
"dense_3/bias/m/Read/ReadVariableOpReadVariableOpdense_3/bias/m*
_output_shapes
: *
dtype0
|
dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_3/kernel/m
u
$dense_3/kernel/m/Read/ReadVariableOpReadVariableOpdense_3/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_2/beta/m

0layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta/m*
_output_shapes
: *
dtype0

layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_2/gamma/m

1layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma/m*
_output_shapes
: *
dtype0
t
dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias/m
m
"dense_2/bias/m/Read/ReadVariableOpReadVariableOpdense_2/bias/m*
_output_shapes
: *
dtype0
|
dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namedense_2/kernel/m
u
$dense_2/kernel/m/Read/ReadVariableOpReadVariableOpdense_2/kernel/m*
_output_shapes

:@ *
dtype0

layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namelayer_normalization_1/beta/m

0layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta/m*
_output_shapes
: *
dtype0

layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namelayer_normalization_1/gamma/m

1layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma/m*
_output_shapes
: *
dtype0
t
dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias/m
m
"dense_1/bias/m/Read/ReadVariableOpReadVariableOpdense_1/bias/m*
_output_shapes
: *
dtype0
|
dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_1/kernel/m
u
$dense_1/kernel/m/Read/ReadVariableOpReadVariableOpdense_1/kernel/m*
_output_shapes

:  *
dtype0

layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization/beta/m

.layer_normalization/beta/m/Read/ReadVariableOpReadVariableOplayer_normalization/beta/m*
_output_shapes
: *
dtype0

layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization/gamma/m

/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOplayer_normalization/gamma/m*
_output_shapes
: *
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
: *
dtype0
x
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel/m
q
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes

: *
dtype0

batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta/m

.batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/m*
_output_shapes
:*
dtype0

batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma/m

/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/m*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
v
regression/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameregression/bias
o
#regression/bias/Read/ReadVariableOpReadVariableOpregression/bias*
_output_shapes
:*
dtype0
~
regression/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameregression/kernel
w
%regression/kernel/Read/ReadVariableOpReadVariableOpregression/kernel*
_output_shapes

: *
dtype0

layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_7/beta

.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
: *
dtype0

layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_7/gamma

/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
: *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
: *
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:  *
dtype0

layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_6/beta

.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
: *
dtype0

layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_6/gamma

/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
: *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:  *
dtype0

layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_5/beta

.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
: *
dtype0

layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_5/gamma

/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:  *
dtype0

layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_4/beta

.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
: *
dtype0

layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_4/gamma

/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
: *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0

layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_3/beta

.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
: *
dtype0

layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_3/gamma

/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:  *
dtype0

layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_2/beta

.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
: *
dtype0

layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_2/gamma

/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@ *
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
: *
dtype0

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:  *
dtype0

layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namelayer_normalization/beta

,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
: *
dtype0

layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namelayer_normalization/gamma

-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Я	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betadense_3/kerneldense_3/biaslayer_normalization_3/gammalayer_normalization_3/betadense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betadense_5/kerneldense_5/biaslayer_normalization_5/gammalayer_normalization_5/betadense_6/kerneldense_6/biaslayer_normalization_6/gammalayer_normalization_6/betadense_7/kerneldense_7/biaslayer_normalization_7/gammalayer_normalization_7/betaregression/kernelregression/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_82676

NoOpNoOp
до
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*о
valueоBџн Bїн
Ж
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
е
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance*
І
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
Џ
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7axis
	8gamma
9beta*
І
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
Џ
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta*

K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
І
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias*
Џ
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_axis
	`gamma
abeta*
І
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
Џ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta*
І
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias*
Г
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta*
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
И
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta*
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
И
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
	Ѓaxis

Єgamma
	Ѕbeta*
Ў
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
Ќkernel
	­bias*
И
Ў	variables
Џtrainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
	Дaxis

Еgamma
	Жbeta*
Ў
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Нkernel
	Оbias*
К
%0
&1
'2
(3
/4
05
86
97
@8
A9
I10
J11
W12
X13
`14
a15
h16
i17
q18
r19
y20
z21
22
23
24
25
26
27
28
29
Є30
Ѕ31
Ќ32
­33
Е34
Ж35
Н36
О37*
Њ
%0
&1
/2
03
84
95
@6
A7
I8
J9
W10
X11
`12
a13
h14
i15
q16
r17
y18
z19
20
21
22
23
24
25
26
27
Є28
Ѕ29
Ќ30
­31
Е32
Ж33
Н34
О35*
* 
Е
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Фtrace_0
Хtrace_1
Цtrace_2
Чtrace_3* 
:
Шtrace_0
Щtrace_1
Ъtrace_2
Ыtrace_3* 
* 
Й
	Ьiter
Эbeta_1
Юbeta_2

Яdecay
аlearning_rate%mф&mх/mц0mч8mш9mщ@mъAmыImьJmэWmюXmя`m№amёhmђimѓqmєrmѕymіzmї	mј	mљ	mњ	mћ	mќ	m§	mў	mџ	Єm	Ѕm	Ќm	­m	Еm	Жm	Нm	Оm%v&v/v0v8v9v@vAvIvJvWvXv`vavhvivqvrvyvzv	v	v	v	v	v 	vЁ	vЂ	vЃ	ЄvЄ	ЅvЅ	ЌvІ	­vЇ	ЕvЈ	ЖvЉ	НvЊ	ОvЋ*

бserving_default* 
 
%0
&1
'2
(3*

%0
&1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

зtrace_0
иtrace_1* 

йtrace_0
кtrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 

тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

чtrace_0* 

шtrace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 

щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

юtrace_0* 

яtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 

№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

ѕtrace_0* 

іtrace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

ќtrace_0* 

§trace_0* 

W0
X1*

W0
X1*
* 

ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

q0
r1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

y0
z1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

trace_0* 

 trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Іtrace_0* 

Їtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

­trace_0* 

Ўtrace_0* 
_Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_5/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_5/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_5/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Лtrace_0* 

Мtrace_0* 
_Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_6/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Є0
Ѕ1*

Є0
Ѕ1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses*

Тtrace_0* 

Уtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_6/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_6/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*

Ќ0
­1*

Ќ0
­1*
* 

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses*

Щtrace_0* 

Ъtrace_0* 
_Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_7/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

Е0
Ж1*

Е0
Ж1*
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
Ў	variables
Џtrainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses*

аtrace_0* 

бtrace_0* 
* 
ke
VARIABLE_VALUElayer_normalization_7/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_7/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*

Н0
О1*

Н0
О1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*

зtrace_0* 

иtrace_0* 
b\
VARIABLE_VALUEregression/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEregression/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19*

й0
к1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

'0
(1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
л	variables
м	keras_api

нtotal

оcount*
M
п	variables
р	keras_api

сtotal

тcount
у
_fn_kwargs*

н0
о1*

л	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

с0
т1*

п	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUEbatch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEbatch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUElayer_normalization/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_4/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_4/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_5/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_5/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_6/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_6/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_7/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_7/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEregression/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEregression/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEbatch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEbatch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUElayer_normalization/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_2/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_3/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEdense_4/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEdense_4/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_4/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_5/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_5/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_5/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_6/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_6/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_6/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEdense_7/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEdense_7/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUElayer_normalization_7/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEregression/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEregression/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
З*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOp%regression/kernel/Read/ReadVariableOp#regression/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/batch_normalization/gamma/m/Read/ReadVariableOp.batch_normalization/beta/m/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp/layer_normalization/gamma/m/Read/ReadVariableOp.layer_normalization/beta/m/Read/ReadVariableOp$dense_1/kernel/m/Read/ReadVariableOp"dense_1/bias/m/Read/ReadVariableOp1layer_normalization_1/gamma/m/Read/ReadVariableOp0layer_normalization_1/beta/m/Read/ReadVariableOp$dense_2/kernel/m/Read/ReadVariableOp"dense_2/bias/m/Read/ReadVariableOp1layer_normalization_2/gamma/m/Read/ReadVariableOp0layer_normalization_2/beta/m/Read/ReadVariableOp$dense_3/kernel/m/Read/ReadVariableOp"dense_3/bias/m/Read/ReadVariableOp1layer_normalization_3/gamma/m/Read/ReadVariableOp0layer_normalization_3/beta/m/Read/ReadVariableOp$dense_4/kernel/m/Read/ReadVariableOp"dense_4/bias/m/Read/ReadVariableOp1layer_normalization_4/gamma/m/Read/ReadVariableOp0layer_normalization_4/beta/m/Read/ReadVariableOp$dense_5/kernel/m/Read/ReadVariableOp"dense_5/bias/m/Read/ReadVariableOp1layer_normalization_5/gamma/m/Read/ReadVariableOp0layer_normalization_5/beta/m/Read/ReadVariableOp$dense_6/kernel/m/Read/ReadVariableOp"dense_6/bias/m/Read/ReadVariableOp1layer_normalization_6/gamma/m/Read/ReadVariableOp0layer_normalization_6/beta/m/Read/ReadVariableOp$dense_7/kernel/m/Read/ReadVariableOp"dense_7/bias/m/Read/ReadVariableOp1layer_normalization_7/gamma/m/Read/ReadVariableOp0layer_normalization_7/beta/m/Read/ReadVariableOp'regression/kernel/m/Read/ReadVariableOp%regression/bias/m/Read/ReadVariableOp/batch_normalization/gamma/v/Read/ReadVariableOp.batch_normalization/beta/v/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOp/layer_normalization/gamma/v/Read/ReadVariableOp.layer_normalization/beta/v/Read/ReadVariableOp$dense_1/kernel/v/Read/ReadVariableOp"dense_1/bias/v/Read/ReadVariableOp1layer_normalization_1/gamma/v/Read/ReadVariableOp0layer_normalization_1/beta/v/Read/ReadVariableOp$dense_2/kernel/v/Read/ReadVariableOp"dense_2/bias/v/Read/ReadVariableOp1layer_normalization_2/gamma/v/Read/ReadVariableOp0layer_normalization_2/beta/v/Read/ReadVariableOp$dense_3/kernel/v/Read/ReadVariableOp"dense_3/bias/v/Read/ReadVariableOp1layer_normalization_3/gamma/v/Read/ReadVariableOp0layer_normalization_3/beta/v/Read/ReadVariableOp$dense_4/kernel/v/Read/ReadVariableOp"dense_4/bias/v/Read/ReadVariableOp1layer_normalization_4/gamma/v/Read/ReadVariableOp0layer_normalization_4/beta/v/Read/ReadVariableOp$dense_5/kernel/v/Read/ReadVariableOp"dense_5/bias/v/Read/ReadVariableOp1layer_normalization_5/gamma/v/Read/ReadVariableOp0layer_normalization_5/beta/v/Read/ReadVariableOp$dense_6/kernel/v/Read/ReadVariableOp"dense_6/bias/v/Read/ReadVariableOp1layer_normalization_6/gamma/v/Read/ReadVariableOp0layer_normalization_6/beta/v/Read/ReadVariableOp$dense_7/kernel/v/Read/ReadVariableOp"dense_7/bias/v/Read/ReadVariableOp1layer_normalization_7/gamma/v/Read/ReadVariableOp0layer_normalization_7/beta/v/Read/ReadVariableOp'regression/kernel/v/Read/ReadVariableOp%regression/bias/v/Read/ReadVariableOpConst*
Tin}
{2y	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_84691
ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betadense_3/kerneldense_3/biaslayer_normalization_3/gammalayer_normalization_3/betadense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betadense_5/kerneldense_5/biaslayer_normalization_5/gammalayer_normalization_5/betadense_6/kerneldense_6/biaslayer_normalization_6/gammalayer_normalization_6/betadense_7/kerneldense_7/biaslayer_normalization_7/gammalayer_normalization_7/betaregression/kernelregression/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountbatch_normalization/gamma/mbatch_normalization/beta/mdense/kernel/mdense/bias/mlayer_normalization/gamma/mlayer_normalization/beta/mdense_1/kernel/mdense_1/bias/mlayer_normalization_1/gamma/mlayer_normalization_1/beta/mdense_2/kernel/mdense_2/bias/mlayer_normalization_2/gamma/mlayer_normalization_2/beta/mdense_3/kernel/mdense_3/bias/mlayer_normalization_3/gamma/mlayer_normalization_3/beta/mdense_4/kernel/mdense_4/bias/mlayer_normalization_4/gamma/mlayer_normalization_4/beta/mdense_5/kernel/mdense_5/bias/mlayer_normalization_5/gamma/mlayer_normalization_5/beta/mdense_6/kernel/mdense_6/bias/mlayer_normalization_6/gamma/mlayer_normalization_6/beta/mdense_7/kernel/mdense_7/bias/mlayer_normalization_7/gamma/mlayer_normalization_7/beta/mregression/kernel/mregression/bias/mbatch_normalization/gamma/vbatch_normalization/beta/vdense/kernel/vdense/bias/vlayer_normalization/gamma/vlayer_normalization/beta/vdense_1/kernel/vdense_1/bias/vlayer_normalization_1/gamma/vlayer_normalization_1/beta/vdense_2/kernel/vdense_2/bias/vlayer_normalization_2/gamma/vlayer_normalization_2/beta/vdense_3/kernel/vdense_3/bias/vlayer_normalization_3/gamma/vlayer_normalization_3/beta/vdense_4/kernel/vdense_4/bias/vlayer_normalization_4/gamma/vlayer_normalization_4/beta/vdense_5/kernel/vdense_5/bias/vlayer_normalization_5/gamma/vlayer_normalization_5/beta/vdense_6/kernel/vdense_6/bias/vlayer_normalization_6/gamma/vlayer_normalization_6/beta/vdense_7/kernel/vdense_7/bias/vlayer_normalization_7/gamma/vlayer_normalization_7/beta/vregression/kernel/vregression/bias/v*
Tin|
z2x*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_85058Љг
њf
П
@__inference_model_layer_call_and_return_conditional_losses_82488
input_1'
batch_normalization_82392:'
batch_normalization_82394:'
batch_normalization_82396:'
batch_normalization_82398:
dense_82401: 
dense_82403: '
layer_normalization_82406: '
layer_normalization_82408: 
dense_1_82411:  
dense_1_82413: )
layer_normalization_1_82416: )
layer_normalization_1_82418: 
dense_2_82422:@ 
dense_2_82424: )
layer_normalization_2_82427: )
layer_normalization_2_82429: 
dense_3_82432:  
dense_3_82434: )
layer_normalization_3_82437: )
layer_normalization_3_82439: 
dense_4_82442:  
dense_4_82444: )
layer_normalization_4_82447: )
layer_normalization_4_82449: 
dense_5_82452:  
dense_5_82454: )
layer_normalization_5_82457: )
layer_normalization_5_82459: 
dense_6_82462:  
dense_6_82464: )
layer_normalization_6_82467: )
layer_normalization_6_82469: 
dense_7_82472:  
dense_7_82474: )
layer_normalization_7_82477: )
layer_normalization_7_82479: "
regression_82482: 
regression_82484:
identityЂ+batch_normalization/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ+layer_normalization/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallЂ-layer_normalization_2/StatefulPartitionedCallЂ-layer_normalization_3/StatefulPartitionedCallЂ-layer_normalization_4/StatefulPartitionedCallЂ-layer_normalization_5/StatefulPartitionedCallЂ-layer_normalization_6/StatefulPartitionedCallЂ-layer_normalization_7/StatefulPartitionedCallЂ"regression/StatefulPartitionedCallз
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_82392batch_normalization_82394batch_normalization_82396batch_normalization_82398*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81170
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82401dense_82403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_81255М
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_82406layer_normalization_82408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_82411dense_1_82413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_81320Ц
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0layer_normalization_1_82416layer_normalization_1_82418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368Ї
concatenate/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_81381
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_82422dense_2_82424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_81394Ц
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0layer_normalization_2_82427layer_normalization_2_82429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_3_82432dense_3_82434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_81459Ц
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0layer_normalization_3_82437layer_normalization_3_82439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507
dense_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_4_82442dense_4_82444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_81524Ц
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0layer_normalization_4_82447layer_normalization_4_82449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_5_82452dense_5_82454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_81589Ц
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0layer_normalization_5_82457layer_normalization_5_82459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_6_82462dense_6_82464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_81654Ц
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_6_82467layer_normalization_6_82469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_7_82472dense_7_82474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_81719Ц
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_7_82477layer_normalization_7_82479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767Ј
"regression/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0regression_82482regression_82484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_regression_layer_call_and_return_conditional_losses_81784z
IdentityIdentity+regression/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall#^regression/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2H
"regression/StatefulPartitionedCall"regression/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ѕf
О
@__inference_model_layer_call_and_return_conditional_losses_82229

inputs'
batch_normalization_82133:'
batch_normalization_82135:'
batch_normalization_82137:'
batch_normalization_82139:
dense_82142: 
dense_82144: '
layer_normalization_82147: '
layer_normalization_82149: 
dense_1_82152:  
dense_1_82154: )
layer_normalization_1_82157: )
layer_normalization_1_82159: 
dense_2_82163:@ 
dense_2_82165: )
layer_normalization_2_82168: )
layer_normalization_2_82170: 
dense_3_82173:  
dense_3_82175: )
layer_normalization_3_82178: )
layer_normalization_3_82180: 
dense_4_82183:  
dense_4_82185: )
layer_normalization_4_82188: )
layer_normalization_4_82190: 
dense_5_82193:  
dense_5_82195: )
layer_normalization_5_82198: )
layer_normalization_5_82200: 
dense_6_82203:  
dense_6_82205: )
layer_normalization_6_82208: )
layer_normalization_6_82210: 
dense_7_82213:  
dense_7_82215: )
layer_normalization_7_82218: )
layer_normalization_7_82220: "
regression_82223: 
regression_82225:
identityЂ+batch_normalization/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ+layer_normalization/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallЂ-layer_normalization_2/StatefulPartitionedCallЂ-layer_normalization_3/StatefulPartitionedCallЂ-layer_normalization_4/StatefulPartitionedCallЂ-layer_normalization_5/StatefulPartitionedCallЂ-layer_normalization_6/StatefulPartitionedCallЂ-layer_normalization_7/StatefulPartitionedCallЂ"regression/StatefulPartitionedCallд
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_82133batch_normalization_82135batch_normalization_82137batch_normalization_82139*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81217
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82142dense_82144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_81255М
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_82147layer_normalization_82149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_82152dense_1_82154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_81320Ц
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0layer_normalization_1_82157layer_normalization_1_82159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368Ї
concatenate/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_81381
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_82163dense_2_82165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_81394Ц
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0layer_normalization_2_82168layer_normalization_2_82170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_3_82173dense_3_82175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_81459Ц
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0layer_normalization_3_82178layer_normalization_3_82180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507
dense_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_4_82183dense_4_82185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_81524Ц
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0layer_normalization_4_82188layer_normalization_4_82190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_5_82193dense_5_82195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_81589Ц
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0layer_normalization_5_82198layer_normalization_5_82200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_6_82203dense_6_82205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_81654Ц
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_6_82208layer_normalization_6_82210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_7_82213dense_7_82215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_81719Ц
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_7_82218layer_normalization_7_82220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767Ј
"regression/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0regression_82223regression_82225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_regression_layer_call_and_return_conditional_losses_81784z
IdentityIdentity+regression/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall#^regression/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2H
"regression/StatefulPartitionedCall"regression/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н

%__inference_dense_layer_call_fn_83719

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_81255o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќм
Њ2
__inference__traced_save_84691
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop0
,savev2_regression_kernel_read_readvariableop.
*savev2_regression_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_batch_normalization_gamma_m_read_readvariableop9
5savev2_batch_normalization_beta_m_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop:
6savev2_layer_normalization_gamma_m_read_readvariableop9
5savev2_layer_normalization_beta_m_read_readvariableop/
+savev2_dense_1_kernel_m_read_readvariableop-
)savev2_dense_1_bias_m_read_readvariableop<
8savev2_layer_normalization_1_gamma_m_read_readvariableop;
7savev2_layer_normalization_1_beta_m_read_readvariableop/
+savev2_dense_2_kernel_m_read_readvariableop-
)savev2_dense_2_bias_m_read_readvariableop<
8savev2_layer_normalization_2_gamma_m_read_readvariableop;
7savev2_layer_normalization_2_beta_m_read_readvariableop/
+savev2_dense_3_kernel_m_read_readvariableop-
)savev2_dense_3_bias_m_read_readvariableop<
8savev2_layer_normalization_3_gamma_m_read_readvariableop;
7savev2_layer_normalization_3_beta_m_read_readvariableop/
+savev2_dense_4_kernel_m_read_readvariableop-
)savev2_dense_4_bias_m_read_readvariableop<
8savev2_layer_normalization_4_gamma_m_read_readvariableop;
7savev2_layer_normalization_4_beta_m_read_readvariableop/
+savev2_dense_5_kernel_m_read_readvariableop-
)savev2_dense_5_bias_m_read_readvariableop<
8savev2_layer_normalization_5_gamma_m_read_readvariableop;
7savev2_layer_normalization_5_beta_m_read_readvariableop/
+savev2_dense_6_kernel_m_read_readvariableop-
)savev2_dense_6_bias_m_read_readvariableop<
8savev2_layer_normalization_6_gamma_m_read_readvariableop;
7savev2_layer_normalization_6_beta_m_read_readvariableop/
+savev2_dense_7_kernel_m_read_readvariableop-
)savev2_dense_7_bias_m_read_readvariableop<
8savev2_layer_normalization_7_gamma_m_read_readvariableop;
7savev2_layer_normalization_7_beta_m_read_readvariableop2
.savev2_regression_kernel_m_read_readvariableop0
,savev2_regression_bias_m_read_readvariableop:
6savev2_batch_normalization_gamma_v_read_readvariableop9
5savev2_batch_normalization_beta_v_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop:
6savev2_layer_normalization_gamma_v_read_readvariableop9
5savev2_layer_normalization_beta_v_read_readvariableop/
+savev2_dense_1_kernel_v_read_readvariableop-
)savev2_dense_1_bias_v_read_readvariableop<
8savev2_layer_normalization_1_gamma_v_read_readvariableop;
7savev2_layer_normalization_1_beta_v_read_readvariableop/
+savev2_dense_2_kernel_v_read_readvariableop-
)savev2_dense_2_bias_v_read_readvariableop<
8savev2_layer_normalization_2_gamma_v_read_readvariableop;
7savev2_layer_normalization_2_beta_v_read_readvariableop/
+savev2_dense_3_kernel_v_read_readvariableop-
)savev2_dense_3_bias_v_read_readvariableop<
8savev2_layer_normalization_3_gamma_v_read_readvariableop;
7savev2_layer_normalization_3_beta_v_read_readvariableop/
+savev2_dense_4_kernel_v_read_readvariableop-
)savev2_dense_4_bias_v_read_readvariableop<
8savev2_layer_normalization_4_gamma_v_read_readvariableop;
7savev2_layer_normalization_4_beta_v_read_readvariableop/
+savev2_dense_5_kernel_v_read_readvariableop-
)savev2_dense_5_bias_v_read_readvariableop<
8savev2_layer_normalization_5_gamma_v_read_readvariableop;
7savev2_layer_normalization_5_beta_v_read_readvariableop/
+savev2_dense_6_kernel_v_read_readvariableop-
)savev2_dense_6_bias_v_read_readvariableop<
8savev2_layer_normalization_6_gamma_v_read_readvariableop;
7savev2_layer_normalization_6_beta_v_read_readvariableop/
+savev2_dense_7_kernel_v_read_readvariableop-
)savev2_dense_7_bias_v_read_readvariableop<
8savev2_layer_normalization_7_gamma_v_read_readvariableop;
7savev2_layer_normalization_7_beta_v_read_readvariableop2
.savev2_regression_kernel_v_read_readvariableop0
,savev2_regression_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: юC
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:x*
dtype0*C
valueCBCxB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHр
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:x*
dtype0*
valueћBјxB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 0
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop6savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableop,savev2_regression_kernel_read_readvariableop*savev2_regression_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_batch_normalization_gamma_m_read_readvariableop5savev2_batch_normalization_beta_m_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop6savev2_layer_normalization_gamma_m_read_readvariableop5savev2_layer_normalization_beta_m_read_readvariableop+savev2_dense_1_kernel_m_read_readvariableop)savev2_dense_1_bias_m_read_readvariableop8savev2_layer_normalization_1_gamma_m_read_readvariableop7savev2_layer_normalization_1_beta_m_read_readvariableop+savev2_dense_2_kernel_m_read_readvariableop)savev2_dense_2_bias_m_read_readvariableop8savev2_layer_normalization_2_gamma_m_read_readvariableop7savev2_layer_normalization_2_beta_m_read_readvariableop+savev2_dense_3_kernel_m_read_readvariableop)savev2_dense_3_bias_m_read_readvariableop8savev2_layer_normalization_3_gamma_m_read_readvariableop7savev2_layer_normalization_3_beta_m_read_readvariableop+savev2_dense_4_kernel_m_read_readvariableop)savev2_dense_4_bias_m_read_readvariableop8savev2_layer_normalization_4_gamma_m_read_readvariableop7savev2_layer_normalization_4_beta_m_read_readvariableop+savev2_dense_5_kernel_m_read_readvariableop)savev2_dense_5_bias_m_read_readvariableop8savev2_layer_normalization_5_gamma_m_read_readvariableop7savev2_layer_normalization_5_beta_m_read_readvariableop+savev2_dense_6_kernel_m_read_readvariableop)savev2_dense_6_bias_m_read_readvariableop8savev2_layer_normalization_6_gamma_m_read_readvariableop7savev2_layer_normalization_6_beta_m_read_readvariableop+savev2_dense_7_kernel_m_read_readvariableop)savev2_dense_7_bias_m_read_readvariableop8savev2_layer_normalization_7_gamma_m_read_readvariableop7savev2_layer_normalization_7_beta_m_read_readvariableop.savev2_regression_kernel_m_read_readvariableop,savev2_regression_bias_m_read_readvariableop6savev2_batch_normalization_gamma_v_read_readvariableop5savev2_batch_normalization_beta_v_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableop6savev2_layer_normalization_gamma_v_read_readvariableop5savev2_layer_normalization_beta_v_read_readvariableop+savev2_dense_1_kernel_v_read_readvariableop)savev2_dense_1_bias_v_read_readvariableop8savev2_layer_normalization_1_gamma_v_read_readvariableop7savev2_layer_normalization_1_beta_v_read_readvariableop+savev2_dense_2_kernel_v_read_readvariableop)savev2_dense_2_bias_v_read_readvariableop8savev2_layer_normalization_2_gamma_v_read_readvariableop7savev2_layer_normalization_2_beta_v_read_readvariableop+savev2_dense_3_kernel_v_read_readvariableop)savev2_dense_3_bias_v_read_readvariableop8savev2_layer_normalization_3_gamma_v_read_readvariableop7savev2_layer_normalization_3_beta_v_read_readvariableop+savev2_dense_4_kernel_v_read_readvariableop)savev2_dense_4_bias_v_read_readvariableop8savev2_layer_normalization_4_gamma_v_read_readvariableop7savev2_layer_normalization_4_beta_v_read_readvariableop+savev2_dense_5_kernel_v_read_readvariableop)savev2_dense_5_bias_v_read_readvariableop8savev2_layer_normalization_5_gamma_v_read_readvariableop7savev2_layer_normalization_5_beta_v_read_readvariableop+savev2_dense_6_kernel_v_read_readvariableop)savev2_dense_6_bias_v_read_readvariableop8savev2_layer_normalization_6_gamma_v_read_readvariableop7savev2_layer_normalization_6_beta_v_read_readvariableop+savev2_dense_7_kernel_v_read_readvariableop)savev2_dense_7_bias_v_read_readvariableop8savev2_layer_normalization_7_gamma_v_read_readvariableop7savev2_layer_normalization_7_beta_v_read_readvariableop.savev2_regression_kernel_v_read_readvariableop,savev2_regression_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes|
z2x	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ћ
_input_shapes
: ::::: : : : :  : : : :@ : : : :  : : : :  : : : :  : : : :  : : : :  : : : : :: : : : : : : : : ::: : : : :  : : : :@ : : : :  : : : :  : : : :  : : : :  : : : :  : : : : :::: : : : :  : : : :@ : : : :  : : : :  : : : :  : : : :  : : : :  : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: :$! 

_output_shapes

:  : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :$% 

_output_shapes

: : &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: : 0

_output_shapes
:: 1

_output_shapes
::$2 

_output_shapes

: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :$6 

_output_shapes

:  : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :$: 

_output_shapes

:@ : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :$> 

_output_shapes

:  : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: :$B 

_output_shapes

:  : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: :$F 

_output_shapes

:  : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: :$J 

_output_shapes

:  : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :$N 

_output_shapes

:  : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :$R 

_output_shapes

: : S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
::$V 

_output_shapes

: : W

_output_shapes
: : X

_output_shapes
: : Y

_output_shapes
: :$Z 

_output_shapes

:  : [

_output_shapes
: : \

_output_shapes
: : ]

_output_shapes
: :$^ 

_output_shapes

:@ : _

_output_shapes
: : `

_output_shapes
: : a

_output_shapes
: :$b 

_output_shapes

:  : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: :$f 

_output_shapes

:  : g

_output_shapes
: : h

_output_shapes
: : i

_output_shapes
: :$j 

_output_shapes

:  : k

_output_shapes
: : l

_output_shapes
: : m

_output_shapes
: :$n 

_output_shapes

:  : o

_output_shapes
: : p

_output_shapes
: : q

_output_shapes
: :$r 

_output_shapes

:  : s

_output_shapes
: : t

_output_shapes
: : u

_output_shapes
: :$v 

_output_shapes

: : w

_output_shapes
::x

_output_shapes
: 
ч
ѓ
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_84291

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_84098

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
л
"
 __inference__wrapped_model_81146
input_1I
;model_batch_normalization_batchnorm_readvariableop_resource:M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:K
=model_batch_normalization_batchnorm_readvariableop_1_resource:K
=model_batch_normalization_batchnorm_readvariableop_2_resource:<
*model_dense_matmul_readvariableop_resource: 9
+model_dense_biasadd_readvariableop_resource: E
7model_layer_normalization_mul_2_readvariableop_resource: C
5model_layer_normalization_add_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource:  ;
-model_dense_1_biasadd_readvariableop_resource: G
9model_layer_normalization_1_mul_2_readvariableop_resource: E
7model_layer_normalization_1_add_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource:@ ;
-model_dense_2_biasadd_readvariableop_resource: G
9model_layer_normalization_2_mul_2_readvariableop_resource: E
7model_layer_normalization_2_add_readvariableop_resource: >
,model_dense_3_matmul_readvariableop_resource:  ;
-model_dense_3_biasadd_readvariableop_resource: G
9model_layer_normalization_3_mul_2_readvariableop_resource: E
7model_layer_normalization_3_add_readvariableop_resource: >
,model_dense_4_matmul_readvariableop_resource:  ;
-model_dense_4_biasadd_readvariableop_resource: G
9model_layer_normalization_4_mul_2_readvariableop_resource: E
7model_layer_normalization_4_add_readvariableop_resource: >
,model_dense_5_matmul_readvariableop_resource:  ;
-model_dense_5_biasadd_readvariableop_resource: G
9model_layer_normalization_5_mul_2_readvariableop_resource: E
7model_layer_normalization_5_add_readvariableop_resource: >
,model_dense_6_matmul_readvariableop_resource:  ;
-model_dense_6_biasadd_readvariableop_resource: G
9model_layer_normalization_6_mul_2_readvariableop_resource: E
7model_layer_normalization_6_add_readvariableop_resource: >
,model_dense_7_matmul_readvariableop_resource:  ;
-model_dense_7_biasadd_readvariableop_resource: G
9model_layer_normalization_7_mul_2_readvariableop_resource: E
7model_layer_normalization_7_add_readvariableop_resource: A
/model_regression_matmul_readvariableop_resource: >
0model_regression_biasadd_readvariableop_resource:
identityЂ2model/batch_normalization/batchnorm/ReadVariableOpЂ4model/batch_normalization/batchnorm/ReadVariableOp_1Ђ4model/batch_normalization/batchnorm/ReadVariableOp_2Ђ6model/batch_normalization/batchnorm/mul/ReadVariableOpЂ"model/dense/BiasAdd/ReadVariableOpЂ!model/dense/MatMul/ReadVariableOpЂ$model/dense_1/BiasAdd/ReadVariableOpЂ#model/dense_1/MatMul/ReadVariableOpЂ$model/dense_2/BiasAdd/ReadVariableOpЂ#model/dense_2/MatMul/ReadVariableOpЂ$model/dense_3/BiasAdd/ReadVariableOpЂ#model/dense_3/MatMul/ReadVariableOpЂ$model/dense_4/BiasAdd/ReadVariableOpЂ#model/dense_4/MatMul/ReadVariableOpЂ$model/dense_5/BiasAdd/ReadVariableOpЂ#model/dense_5/MatMul/ReadVariableOpЂ$model/dense_6/BiasAdd/ReadVariableOpЂ#model/dense_6/MatMul/ReadVariableOpЂ$model/dense_7/BiasAdd/ReadVariableOpЂ#model/dense_7/MatMul/ReadVariableOpЂ,model/layer_normalization/add/ReadVariableOpЂ.model/layer_normalization/mul_2/ReadVariableOpЂ.model/layer_normalization_1/add/ReadVariableOpЂ0model/layer_normalization_1/mul_2/ReadVariableOpЂ.model/layer_normalization_2/add/ReadVariableOpЂ0model/layer_normalization_2/mul_2/ReadVariableOpЂ.model/layer_normalization_3/add/ReadVariableOpЂ0model/layer_normalization_3/mul_2/ReadVariableOpЂ.model/layer_normalization_4/add/ReadVariableOpЂ0model/layer_normalization_4/mul_2/ReadVariableOpЂ.model/layer_normalization_5/add/ReadVariableOpЂ0model/layer_normalization_5/mul_2/ReadVariableOpЂ.model/layer_normalization_6/add/ReadVariableOpЂ0model/layer_normalization_6/mul_2/ReadVariableOpЂ.model/layer_normalization_7/add/ReadVariableOpЂ0model/layer_normalization_7/mul_2/ReadVariableOpЂ'model/regression/BiasAdd/ReadVariableOpЂ&model/regression/MatMul/ReadVariableOpЊ
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Х
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:В
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Т
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
)model/batch_normalization/batchnorm/mul_1Mulinput_1+model/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Р
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:Ў
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Р
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Р
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
model/dense/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ h
model/dense/TanhTanhmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
model/layer_normalization/ShapeShapemodel/dense/Tanh:y:0*
T0*
_output_shapes
:w
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ё
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization/mul_1Mul*model/layer_normalization/mul_1/x:output:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: k
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :k
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0!model/layer_normalization/mul:z:0#model/layer_normalization/mul_1:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ў
!model/layer_normalization/ReshapeReshapemodel/dense/Tanh:y:00model/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ ~
%model/layer_normalization/ones/packedPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:i
$model/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Г
model/layer_normalization/onesFill.model/layer_normalization/ones/packed:output:0-model/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
&model/layer_normalization/zeros/packedPack!model/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:j
%model/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ж
model/layer_normalization/zerosFill/model/layer_normalization/zeros/packed:output:0.model/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџb
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB П
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/ones:output:0(model/layer_normalization/zeros:output:0(model/layer_normalization/Const:output:0*model/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:К
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization/mul_2/ReadVariableOpReadVariableOp7model_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0О
model/layer_normalization/mul_2Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes
: *
dtype0Г
model/layer_normalization/addAddV2#model/layer_normalization/mul_2:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0 
model/dense_1/MatMulMatMul!model/layer_normalization/add:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_1/TanhTanhmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_1/ShapeShapemodel/dense_1/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_1/strided_sliceStridedSlice*model/layer_normalization_1/Shape:output:08model/layer_normalization_1/strided_slice/stack:output:0:model/layer_normalization_1/strided_slice/stack_1:output:0:model/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_1/mulMul*model/layer_normalization_1/mul/x:output:02model/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_1/strided_slice_1StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_1/stack:output:0<model/layer_normalization_1/strided_slice_1/stack_1:output:0<model/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_1/mul_1Mul,model/layer_normalization_1/mul_1/x:output:04model/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_1/Reshape/shapePack4model/layer_normalization_1/Reshape/shape/0:output:0#model/layer_normalization_1/mul:z:0%model/layer_normalization_1/mul_1:z:04model/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_1/ReshapeReshapemodel/dense_1/Tanh:y:02model/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_1/ones/packedPack#model/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_1/onesFill0model/layer_normalization_1/ones/packed:output:0/model/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_1/zeros/packedPack#model/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_1/zerosFill1model/layer_normalization_1/zeros/packed:output:00model/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_1/Reshape:output:0)model/layer_normalization_1/ones:output:0*model/layer_normalization_1/zeros:output:0*model/layer_normalization_1/Const:output:0,model/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_1/Reshape_1Reshape0model/layer_normalization_1/FusedBatchNormV3:y:0*model/layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_1/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_1/mul_2Mul.model/layer_normalization_1/Reshape_1:output:08model/layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_1/add/ReadVariableOpReadVariableOp7model_layer_normalization_1_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_1/addAddV2%model/layer_normalization_1/mul_2:z:06model/layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Я
model/concatenate/concatConcatV2!model/layer_normalization/add:z:0#model/layer_normalization_1/add:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0 
model/dense_2/MatMulMatMul!model/concatenate/concat:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_2/TanhTanhmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_2/ShapeShapemodel/dense_2/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_2/strided_sliceStridedSlice*model/layer_normalization_2/Shape:output:08model/layer_normalization_2/strided_slice/stack:output:0:model/layer_normalization_2/strided_slice/stack_1:output:0:model/layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_2/mulMul*model/layer_normalization_2/mul/x:output:02model/layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_2/strided_slice_1StridedSlice*model/layer_normalization_2/Shape:output:0:model/layer_normalization_2/strided_slice_1/stack:output:0<model/layer_normalization_2/strided_slice_1/stack_1:output:0<model/layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_2/mul_1Mul,model/layer_normalization_2/mul_1/x:output:04model/layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_2/Reshape/shapePack4model/layer_normalization_2/Reshape/shape/0:output:0#model/layer_normalization_2/mul:z:0%model/layer_normalization_2/mul_1:z:04model/layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_2/ReshapeReshapemodel/dense_2/Tanh:y:02model/layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_2/ones/packedPack#model/layer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_2/onesFill0model/layer_normalization_2/ones/packed:output:0/model/layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_2/zeros/packedPack#model/layer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_2/zerosFill1model/layer_normalization_2/zeros/packed:output:00model/layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_2/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_2/Reshape:output:0)model/layer_normalization_2/ones:output:0*model/layer_normalization_2/zeros:output:0*model/layer_normalization_2/Const:output:0,model/layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_2/Reshape_1Reshape0model/layer_normalization_2/FusedBatchNormV3:y:0*model/layer_normalization_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_2/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_2/mul_2Mul.model/layer_normalization_2/Reshape_1:output:08model/layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_2/add/ReadVariableOpReadVariableOp7model_layer_normalization_2_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_2/addAddV2%model/layer_normalization_2/mul_2:z:06model/layer_normalization_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
model/dense_3/MatMulMatMul#model/layer_normalization_2/add:z:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_3/TanhTanhmodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_3/ShapeShapemodel/dense_3/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_3/strided_sliceStridedSlice*model/layer_normalization_3/Shape:output:08model/layer_normalization_3/strided_slice/stack:output:0:model/layer_normalization_3/strided_slice/stack_1:output:0:model/layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_3/mulMul*model/layer_normalization_3/mul/x:output:02model/layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_3/strided_slice_1StridedSlice*model/layer_normalization_3/Shape:output:0:model/layer_normalization_3/strided_slice_1/stack:output:0<model/layer_normalization_3/strided_slice_1/stack_1:output:0<model/layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_3/mul_1Mul,model/layer_normalization_3/mul_1/x:output:04model/layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_3/Reshape/shapePack4model/layer_normalization_3/Reshape/shape/0:output:0#model/layer_normalization_3/mul:z:0%model/layer_normalization_3/mul_1:z:04model/layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_3/ReshapeReshapemodel/dense_3/Tanh:y:02model/layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_3/ones/packedPack#model/layer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_3/onesFill0model/layer_normalization_3/ones/packed:output:0/model/layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_3/zeros/packedPack#model/layer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_3/zerosFill1model/layer_normalization_3/zeros/packed:output:00model/layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_3/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_3/Reshape:output:0)model/layer_normalization_3/ones:output:0*model/layer_normalization_3/zeros:output:0*model/layer_normalization_3/Const:output:0,model/layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_3/Reshape_1Reshape0model/layer_normalization_3/FusedBatchNormV3:y:0*model/layer_normalization_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_3/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_3_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_3/mul_2Mul.model/layer_normalization_3/Reshape_1:output:08model/layer_normalization_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_3/add/ReadVariableOpReadVariableOp7model_layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_3/addAddV2%model/layer_normalization_3/mul_2:z:06model/layer_normalization_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
model/dense_4/MatMulMatMul#model/layer_normalization_3/add:z:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_4/TanhTanhmodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_4/ShapeShapemodel/dense_4/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_4/strided_sliceStridedSlice*model/layer_normalization_4/Shape:output:08model/layer_normalization_4/strided_slice/stack:output:0:model/layer_normalization_4/strided_slice/stack_1:output:0:model/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_4/mulMul*model/layer_normalization_4/mul/x:output:02model/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_4/strided_slice_1StridedSlice*model/layer_normalization_4/Shape:output:0:model/layer_normalization_4/strided_slice_1/stack:output:0<model/layer_normalization_4/strided_slice_1/stack_1:output:0<model/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_4/mul_1Mul,model/layer_normalization_4/mul_1/x:output:04model/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_4/Reshape/shapePack4model/layer_normalization_4/Reshape/shape/0:output:0#model/layer_normalization_4/mul:z:0%model/layer_normalization_4/mul_1:z:04model/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_4/ReshapeReshapemodel/dense_4/Tanh:y:02model/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_4/ones/packedPack#model/layer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_4/onesFill0model/layer_normalization_4/ones/packed:output:0/model/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_4/zeros/packedPack#model/layer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_4/zerosFill1model/layer_normalization_4/zeros/packed:output:00model/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_4/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_4/Reshape:output:0)model/layer_normalization_4/ones:output:0*model/layer_normalization_4/zeros:output:0*model/layer_normalization_4/Const:output:0,model/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_4/Reshape_1Reshape0model/layer_normalization_4/FusedBatchNormV3:y:0*model/layer_normalization_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_4/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_4/mul_2Mul.model/layer_normalization_4/Reshape_1:output:08model/layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_4/add/ReadVariableOpReadVariableOp7model_layer_normalization_4_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_4/addAddV2%model/layer_normalization_4/mul_2:z:06model/layer_normalization_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
model/dense_5/MatMulMatMul#model/layer_normalization_4/add:z:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_5/TanhTanhmodel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_5/ShapeShapemodel/dense_5/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_5/strided_sliceStridedSlice*model/layer_normalization_5/Shape:output:08model/layer_normalization_5/strided_slice/stack:output:0:model/layer_normalization_5/strided_slice/stack_1:output:0:model/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_5/mulMul*model/layer_normalization_5/mul/x:output:02model/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_5/strided_slice_1StridedSlice*model/layer_normalization_5/Shape:output:0:model/layer_normalization_5/strided_slice_1/stack:output:0<model/layer_normalization_5/strided_slice_1/stack_1:output:0<model/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_5/mul_1Mul,model/layer_normalization_5/mul_1/x:output:04model/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_5/Reshape/shapePack4model/layer_normalization_5/Reshape/shape/0:output:0#model/layer_normalization_5/mul:z:0%model/layer_normalization_5/mul_1:z:04model/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_5/ReshapeReshapemodel/dense_5/Tanh:y:02model/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_5/ones/packedPack#model/layer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_5/onesFill0model/layer_normalization_5/ones/packed:output:0/model/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_5/zeros/packedPack#model/layer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_5/zerosFill1model/layer_normalization_5/zeros/packed:output:00model/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_5/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_5/Reshape:output:0)model/layer_normalization_5/ones:output:0*model/layer_normalization_5/zeros:output:0*model/layer_normalization_5/Const:output:0,model/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_5/Reshape_1Reshape0model/layer_normalization_5/FusedBatchNormV3:y:0*model/layer_normalization_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_5/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_5/mul_2Mul.model/layer_normalization_5/Reshape_1:output:08model/layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_5/add/ReadVariableOpReadVariableOp7model_layer_normalization_5_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_5/addAddV2%model/layer_normalization_5/mul_2:z:06model/layer_normalization_5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
model/dense_6/MatMulMatMul#model/layer_normalization_5/add:z:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_6/TanhTanhmodel/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_6/ShapeShapemodel/dense_6/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_6/strided_sliceStridedSlice*model/layer_normalization_6/Shape:output:08model/layer_normalization_6/strided_slice/stack:output:0:model/layer_normalization_6/strided_slice/stack_1:output:0:model/layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_6/mulMul*model/layer_normalization_6/mul/x:output:02model/layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_6/strided_slice_1StridedSlice*model/layer_normalization_6/Shape:output:0:model/layer_normalization_6/strided_slice_1/stack:output:0<model/layer_normalization_6/strided_slice_1/stack_1:output:0<model/layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_6/mul_1Mul,model/layer_normalization_6/mul_1/x:output:04model/layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_6/Reshape/shapePack4model/layer_normalization_6/Reshape/shape/0:output:0#model/layer_normalization_6/mul:z:0%model/layer_normalization_6/mul_1:z:04model/layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_6/ReshapeReshapemodel/dense_6/Tanh:y:02model/layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_6/ones/packedPack#model/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_6/onesFill0model/layer_normalization_6/ones/packed:output:0/model/layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_6/zeros/packedPack#model/layer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_6/zerosFill1model/layer_normalization_6/zeros/packed:output:00model/layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_6/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_6/Reshape:output:0)model/layer_normalization_6/ones:output:0*model/layer_normalization_6/zeros:output:0*model/layer_normalization_6/Const:output:0,model/layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_6/Reshape_1Reshape0model/layer_normalization_6/FusedBatchNormV3:y:0*model/layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_6/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_6/mul_2Mul.model/layer_normalization_6/Reshape_1:output:08model/layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_6/add/ReadVariableOpReadVariableOp7model_layer_normalization_6_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_6/addAddV2%model/layer_normalization_6/mul_2:z:06model/layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
#model/dense_7/MatMul/ReadVariableOpReadVariableOp,model_dense_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ђ
model/dense_7/MatMulMatMul#model/layer_normalization_6/add:z:0+model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_7/BiasAddBiasAddmodel/dense_7/MatMul:product:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ l
model/dense_7/TanhTanhmodel/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ g
!model/layer_normalization_7/ShapeShapemodel/dense_7/Tanh:y:0*
T0*
_output_shapes
:y
/model/layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model/layer_normalization_7/strided_sliceStridedSlice*model/layer_normalization_7/Shape:output:08model/layer_normalization_7/strided_slice/stack:output:0:model/layer_normalization_7/strided_slice/stack_1:output:0:model/layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :Ї
model/layer_normalization_7/mulMul*model/layer_normalization_7/mul/x:output:02model/layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+model/layer_normalization_7/strided_slice_1StridedSlice*model/layer_normalization_7/Shape:output:0:model/layer_normalization_7/strided_slice_1/stack:output:0<model/layer_normalization_7/strided_slice_1/stack_1:output:0<model/layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :­
!model/layer_normalization_7/mul_1Mul,model/layer_normalization_7/mul_1/x:output:04model/layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
)model/layer_normalization_7/Reshape/shapePack4model/layer_normalization_7/Reshape/shape/0:output:0#model/layer_normalization_7/mul:z:0%model/layer_normalization_7/mul_1:z:04model/layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Д
#model/layer_normalization_7/ReshapeReshapemodel/dense_7/Tanh:y:02model/layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
'model/layer_normalization_7/ones/packedPack#model/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Й
 model/layer_normalization_7/onesFill0model/layer_normalization_7/ones/packed:output:0/model/layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
(model/layer_normalization_7/zeros/packedPack#model/layer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    М
!model/layer_normalization_7/zerosFill1model/layer_normalization_7/zeros/packed:output:00model/layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџd
!model/layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ы
,model/layer_normalization_7/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_7/Reshape:output:0)model/layer_normalization_7/ones:output:0*model/layer_normalization_7/zeros:output:0*model/layer_normalization_7/Const:output:0,model/layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Р
%model/layer_normalization_7/Reshape_1Reshape0model/layer_normalization_7/FusedBatchNormV3:y:0*model/layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ І
0model/layer_normalization_7/mul_2/ReadVariableOpReadVariableOp9model_layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ф
!model/layer_normalization_7/mul_2Mul.model/layer_normalization_7/Reshape_1:output:08model/layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
.model/layer_normalization_7/add/ReadVariableOpReadVariableOp7model_layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0Й
model/layer_normalization_7/addAddV2%model/layer_normalization_7/mul_2:z:06model/layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&model/regression/MatMul/ReadVariableOpReadVariableOp/model_regression_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
model/regression/MatMulMatMul#model/layer_normalization_7/add:z:0.model/regression/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/regression/BiasAdd/ReadVariableOpReadVariableOp0model_regression_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/regression/BiasAddBiasAdd!model/regression/MatMul:product:0/model/regression/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
model/regression/TanhTanh!model/regression/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitymodel/regression/Tanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџѕ
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp%^model/dense_7/BiasAdd/ReadVariableOp$^model/dense_7/MatMul/ReadVariableOp-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_2/ReadVariableOp/^model/layer_normalization_1/add/ReadVariableOp1^model/layer_normalization_1/mul_2/ReadVariableOp/^model/layer_normalization_2/add/ReadVariableOp1^model/layer_normalization_2/mul_2/ReadVariableOp/^model/layer_normalization_3/add/ReadVariableOp1^model/layer_normalization_3/mul_2/ReadVariableOp/^model/layer_normalization_4/add/ReadVariableOp1^model/layer_normalization_4/mul_2/ReadVariableOp/^model/layer_normalization_5/add/ReadVariableOp1^model/layer_normalization_5/mul_2/ReadVariableOp/^model/layer_normalization_6/add/ReadVariableOp1^model/layer_normalization_6/mul_2/ReadVariableOp/^model/layer_normalization_7/add/ReadVariableOp1^model/layer_normalization_7/mul_2/ReadVariableOp(^model/regression/BiasAdd/ReadVariableOp'^model/regression/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2L
$model/dense_7/BiasAdd/ReadVariableOp$model/dense_7/BiasAdd/ReadVariableOp2J
#model/dense_7/MatMul/ReadVariableOp#model/dense_7/MatMul/ReadVariableOp2\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_2/ReadVariableOp.model/layer_normalization/mul_2/ReadVariableOp2`
.model/layer_normalization_1/add/ReadVariableOp.model/layer_normalization_1/add/ReadVariableOp2d
0model/layer_normalization_1/mul_2/ReadVariableOp0model/layer_normalization_1/mul_2/ReadVariableOp2`
.model/layer_normalization_2/add/ReadVariableOp.model/layer_normalization_2/add/ReadVariableOp2d
0model/layer_normalization_2/mul_2/ReadVariableOp0model/layer_normalization_2/mul_2/ReadVariableOp2`
.model/layer_normalization_3/add/ReadVariableOp.model/layer_normalization_3/add/ReadVariableOp2d
0model/layer_normalization_3/mul_2/ReadVariableOp0model/layer_normalization_3/mul_2/ReadVariableOp2`
.model/layer_normalization_4/add/ReadVariableOp.model/layer_normalization_4/add/ReadVariableOp2d
0model/layer_normalization_4/mul_2/ReadVariableOp0model/layer_normalization_4/mul_2/ReadVariableOp2`
.model/layer_normalization_5/add/ReadVariableOp.model/layer_normalization_5/add/ReadVariableOp2d
0model/layer_normalization_5/mul_2/ReadVariableOp0model/layer_normalization_5/mul_2/ReadVariableOp2`
.model/layer_normalization_6/add/ReadVariableOp.model/layer_normalization_6/add/ReadVariableOp2d
0model/layer_normalization_6/mul_2/ReadVariableOp0model/layer_normalization_6/mul_2/ReadVariableOp2`
.model/layer_normalization_7/add/ReadVariableOp.model/layer_normalization_7/add/ReadVariableOp2d
0model/layer_normalization_7/mul_2/ReadVariableOp0model/layer_normalization_7/mul_2/ReadVariableOp2R
'model/regression/BiasAdd/ReadVariableOp'model/regression/BiasAdd/ReadVariableOp2P
&model/regression/MatMul/ReadVariableOp&model/regression/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
С

'__inference_dense_2_layer_call_fn_83874

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_81394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
С

'__inference_dense_7_layer_call_fn_84229

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_81719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_3_layer_call_and_return_conditional_losses_81459

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
З
p
F__inference_concatenate_layer_call_and_return_conditional_losses_81381

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


і
E__inference_regression_layer_call_and_return_conditional_losses_84311

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_81524

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_2_layer_call_fn_83894

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
їf
О
@__inference_model_layer_call_and_return_conditional_losses_81791

inputs'
batch_normalization_81235:'
batch_normalization_81237:'
batch_normalization_81239:'
batch_normalization_81241:
dense_81256: 
dense_81258: '
layer_normalization_81304: '
layer_normalization_81306: 
dense_1_81321:  
dense_1_81323: )
layer_normalization_1_81369: )
layer_normalization_1_81371: 
dense_2_81395:@ 
dense_2_81397: )
layer_normalization_2_81443: )
layer_normalization_2_81445: 
dense_3_81460:  
dense_3_81462: )
layer_normalization_3_81508: )
layer_normalization_3_81510: 
dense_4_81525:  
dense_4_81527: )
layer_normalization_4_81573: )
layer_normalization_4_81575: 
dense_5_81590:  
dense_5_81592: )
layer_normalization_5_81638: )
layer_normalization_5_81640: 
dense_6_81655:  
dense_6_81657: )
layer_normalization_6_81703: )
layer_normalization_6_81705: 
dense_7_81720:  
dense_7_81722: )
layer_normalization_7_81768: )
layer_normalization_7_81770: "
regression_81785: 
regression_81787:
identityЂ+batch_normalization/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ+layer_normalization/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallЂ-layer_normalization_2/StatefulPartitionedCallЂ-layer_normalization_3/StatefulPartitionedCallЂ-layer_normalization_4/StatefulPartitionedCallЂ-layer_normalization_5/StatefulPartitionedCallЂ-layer_normalization_6/StatefulPartitionedCallЂ-layer_normalization_7/StatefulPartitionedCallЂ"regression/StatefulPartitionedCallж
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_81235batch_normalization_81237batch_normalization_81239batch_normalization_81241*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81170
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_81256dense_81258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_81255М
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_81304layer_normalization_81306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_81321dense_1_81323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_81320Ц
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0layer_normalization_1_81369layer_normalization_1_81371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368Ї
concatenate/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_81381
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_81395dense_2_81397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_81394Ц
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0layer_normalization_2_81443layer_normalization_2_81445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_3_81460dense_3_81462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_81459Ц
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0layer_normalization_3_81508layer_normalization_3_81510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507
dense_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_4_81525dense_4_81527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_81524Ц
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0layer_normalization_4_81573layer_normalization_4_81575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_5_81590dense_5_81592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_81589Ц
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0layer_normalization_5_81638layer_normalization_5_81640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_6_81655dense_6_81657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_81654Ц
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_6_81703layer_normalization_6_81705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_7_81720dense_7_81722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_81719Ц
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_7_81768layer_normalization_7_81770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767Ј
"regression/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0regression_81785regression_81787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_regression_layer_call_and_return_conditional_losses_81784z
IdentityIdentity+regression/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall#^regression/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2H
"regression/StatefulPartitionedCall"regression/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С

'__inference_dense_6_layer_call_fn_84158

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_81654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_4_layer_call_fn_84036

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
х
ё
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_6_layer_call_and_return_conditional_losses_81654

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_2_layer_call_and_return_conditional_losses_83885

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е

3__inference_layer_normalization_layer_call_fn_83739

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ё
@__inference_dense_layer_call_and_return_conditional_losses_83730

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
ё
N__inference_layer_normalization_layer_call_and_return_conditional_losses_83781

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ѕ
Ю
3__inference_batch_normalization_layer_call_fn_83643

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
%
ч
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81217

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_84220

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
С

'__inference_dense_3_layer_call_fn_83945

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_81459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_1_layer_call_and_return_conditional_losses_81320

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Б
б
%__inference_model_layer_call_fn_82838

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27:  

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_82229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С

'__inference_dense_5_layer_call_fn_84087

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_81589o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_3_layer_call_fn_83965

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_84027

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_2_layer_call_and_return_conditional_losses_81394

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


ѓ
B__inference_dense_7_layer_call_and_return_conditional_losses_81719

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
уй
§I
!__inference__traced_restore_85058
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance:1
assignvariableop_4_dense_kernel: +
assignvariableop_5_dense_bias: :
,assignvariableop_6_layer_normalization_gamma: 9
+assignvariableop_7_layer_normalization_beta: 3
!assignvariableop_8_dense_1_kernel:  -
assignvariableop_9_dense_1_bias: =
/assignvariableop_10_layer_normalization_1_gamma: <
.assignvariableop_11_layer_normalization_1_beta: 4
"assignvariableop_12_dense_2_kernel:@ .
 assignvariableop_13_dense_2_bias: =
/assignvariableop_14_layer_normalization_2_gamma: <
.assignvariableop_15_layer_normalization_2_beta: 4
"assignvariableop_16_dense_3_kernel:  .
 assignvariableop_17_dense_3_bias: =
/assignvariableop_18_layer_normalization_3_gamma: <
.assignvariableop_19_layer_normalization_3_beta: 4
"assignvariableop_20_dense_4_kernel:  .
 assignvariableop_21_dense_4_bias: =
/assignvariableop_22_layer_normalization_4_gamma: <
.assignvariableop_23_layer_normalization_4_beta: 4
"assignvariableop_24_dense_5_kernel:  .
 assignvariableop_25_dense_5_bias: =
/assignvariableop_26_layer_normalization_5_gamma: <
.assignvariableop_27_layer_normalization_5_beta: 4
"assignvariableop_28_dense_6_kernel:  .
 assignvariableop_29_dense_6_bias: =
/assignvariableop_30_layer_normalization_6_gamma: <
.assignvariableop_31_layer_normalization_6_beta: 4
"assignvariableop_32_dense_7_kernel:  .
 assignvariableop_33_dense_7_bias: =
/assignvariableop_34_layer_normalization_7_gamma: <
.assignvariableop_35_layer_normalization_7_beta: 7
%assignvariableop_36_regression_kernel: 1
#assignvariableop_37_regression_bias:'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: #
assignvariableop_45_total: #
assignvariableop_46_count: =
/assignvariableop_47_batch_normalization_gamma_m:<
.assignvariableop_48_batch_normalization_beta_m:4
"assignvariableop_49_dense_kernel_m: .
 assignvariableop_50_dense_bias_m: =
/assignvariableop_51_layer_normalization_gamma_m: <
.assignvariableop_52_layer_normalization_beta_m: 6
$assignvariableop_53_dense_1_kernel_m:  0
"assignvariableop_54_dense_1_bias_m: ?
1assignvariableop_55_layer_normalization_1_gamma_m: >
0assignvariableop_56_layer_normalization_1_beta_m: 6
$assignvariableop_57_dense_2_kernel_m:@ 0
"assignvariableop_58_dense_2_bias_m: ?
1assignvariableop_59_layer_normalization_2_gamma_m: >
0assignvariableop_60_layer_normalization_2_beta_m: 6
$assignvariableop_61_dense_3_kernel_m:  0
"assignvariableop_62_dense_3_bias_m: ?
1assignvariableop_63_layer_normalization_3_gamma_m: >
0assignvariableop_64_layer_normalization_3_beta_m: 6
$assignvariableop_65_dense_4_kernel_m:  0
"assignvariableop_66_dense_4_bias_m: ?
1assignvariableop_67_layer_normalization_4_gamma_m: >
0assignvariableop_68_layer_normalization_4_beta_m: 6
$assignvariableop_69_dense_5_kernel_m:  0
"assignvariableop_70_dense_5_bias_m: ?
1assignvariableop_71_layer_normalization_5_gamma_m: >
0assignvariableop_72_layer_normalization_5_beta_m: 6
$assignvariableop_73_dense_6_kernel_m:  0
"assignvariableop_74_dense_6_bias_m: ?
1assignvariableop_75_layer_normalization_6_gamma_m: >
0assignvariableop_76_layer_normalization_6_beta_m: 6
$assignvariableop_77_dense_7_kernel_m:  0
"assignvariableop_78_dense_7_bias_m: ?
1assignvariableop_79_layer_normalization_7_gamma_m: >
0assignvariableop_80_layer_normalization_7_beta_m: 9
'assignvariableop_81_regression_kernel_m: 3
%assignvariableop_82_regression_bias_m:=
/assignvariableop_83_batch_normalization_gamma_v:<
.assignvariableop_84_batch_normalization_beta_v:4
"assignvariableop_85_dense_kernel_v: .
 assignvariableop_86_dense_bias_v: =
/assignvariableop_87_layer_normalization_gamma_v: <
.assignvariableop_88_layer_normalization_beta_v: 6
$assignvariableop_89_dense_1_kernel_v:  0
"assignvariableop_90_dense_1_bias_v: ?
1assignvariableop_91_layer_normalization_1_gamma_v: >
0assignvariableop_92_layer_normalization_1_beta_v: 6
$assignvariableop_93_dense_2_kernel_v:@ 0
"assignvariableop_94_dense_2_bias_v: ?
1assignvariableop_95_layer_normalization_2_gamma_v: >
0assignvariableop_96_layer_normalization_2_beta_v: 6
$assignvariableop_97_dense_3_kernel_v:  0
"assignvariableop_98_dense_3_bias_v: ?
1assignvariableop_99_layer_normalization_3_gamma_v: ?
1assignvariableop_100_layer_normalization_3_beta_v: 7
%assignvariableop_101_dense_4_kernel_v:  1
#assignvariableop_102_dense_4_bias_v: @
2assignvariableop_103_layer_normalization_4_gamma_v: ?
1assignvariableop_104_layer_normalization_4_beta_v: 7
%assignvariableop_105_dense_5_kernel_v:  1
#assignvariableop_106_dense_5_bias_v: @
2assignvariableop_107_layer_normalization_5_gamma_v: ?
1assignvariableop_108_layer_normalization_5_beta_v: 7
%assignvariableop_109_dense_6_kernel_v:  1
#assignvariableop_110_dense_6_bias_v: @
2assignvariableop_111_layer_normalization_6_gamma_v: ?
1assignvariableop_112_layer_normalization_6_beta_v: 7
%assignvariableop_113_dense_7_kernel_v:  1
#assignvariableop_114_dense_7_bias_v: @
2assignvariableop_115_layer_normalization_7_gamma_v: ?
1assignvariableop_116_layer_normalization_7_beta_v: :
(assignvariableop_117_regression_kernel_v: 4
&assignvariableop_118_regression_bias_v:
identity_120ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_103ЂAssignVariableOp_104ЂAssignVariableOp_105ЂAssignVariableOp_106ЂAssignVariableOp_107ЂAssignVariableOp_108ЂAssignVariableOp_109ЂAssignVariableOp_11ЂAssignVariableOp_110ЂAssignVariableOp_111ЂAssignVariableOp_112ЂAssignVariableOp_113ЂAssignVariableOp_114ЂAssignVariableOp_115ЂAssignVariableOp_116ЂAssignVariableOp_117ЂAssignVariableOp_118ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99ёC
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:x*
dtype0*C
valueCBCxB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHу
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:x*
dtype0*
valueћBјxB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B њ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*і
_output_shapesу
р::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes|
z2x	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_layer_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_layer_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_layer_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp.assignvariableop_15_layer_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_layer_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_layer_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_22AssignVariableOp/assignvariableop_22_layer_normalization_4_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp.assignvariableop_23_layer_normalization_4_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_layer_normalization_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp.assignvariableop_27_layer_normalization_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_6_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_6_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_30AssignVariableOp/assignvariableop_30_layer_normalization_6_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp.assignvariableop_31_layer_normalization_6_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_7_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_7_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_34AssignVariableOp/assignvariableop_34_layer_normalization_7_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp.assignvariableop_35_layer_normalization_7_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_regression_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp#assignvariableop_37_regression_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_47AssignVariableOp/assignvariableop_47_batch_normalization_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp.assignvariableop_48_batch_normalization_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp"assignvariableop_49_dense_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp assignvariableop_50_dense_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_51AssignVariableOp/assignvariableop_51_layer_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp.assignvariableop_52_layer_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp$assignvariableop_53_dense_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp"assignvariableop_54_dense_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_55AssignVariableOp1assignvariableop_55_layer_normalization_1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_56AssignVariableOp0assignvariableop_56_layer_normalization_1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_59AssignVariableOp1assignvariableop_59_layer_normalization_2_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_60AssignVariableOp0assignvariableop_60_layer_normalization_2_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp$assignvariableop_61_dense_3_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp"assignvariableop_62_dense_3_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_63AssignVariableOp1assignvariableop_63_layer_normalization_3_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_64AssignVariableOp0assignvariableop_64_layer_normalization_3_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp$assignvariableop_65_dense_4_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp"assignvariableop_66_dense_4_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_67AssignVariableOp1assignvariableop_67_layer_normalization_4_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_68AssignVariableOp0assignvariableop_68_layer_normalization_4_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp$assignvariableop_69_dense_5_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp"assignvariableop_70_dense_5_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_71AssignVariableOp1assignvariableop_71_layer_normalization_5_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_72AssignVariableOp0assignvariableop_72_layer_normalization_5_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp$assignvariableop_73_dense_6_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp"assignvariableop_74_dense_6_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_75AssignVariableOp1assignvariableop_75_layer_normalization_6_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_76AssignVariableOp0assignvariableop_76_layer_normalization_6_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp$assignvariableop_77_dense_7_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp"assignvariableop_78_dense_7_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_79AssignVariableOp1assignvariableop_79_layer_normalization_7_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_80AssignVariableOp0assignvariableop_80_layer_normalization_7_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp'assignvariableop_81_regression_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp%assignvariableop_82_regression_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_83AssignVariableOp/assignvariableop_83_batch_normalization_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp.assignvariableop_84_batch_normalization_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp"assignvariableop_85_dense_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp assignvariableop_86_dense_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_87AssignVariableOp/assignvariableop_87_layer_normalization_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp.assignvariableop_88_layer_normalization_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp$assignvariableop_89_dense_1_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp"assignvariableop_90_dense_1_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_91AssignVariableOp1assignvariableop_91_layer_normalization_1_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_92AssignVariableOp0assignvariableop_92_layer_normalization_1_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp$assignvariableop_93_dense_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp"assignvariableop_94_dense_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_95AssignVariableOp1assignvariableop_95_layer_normalization_2_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_96AssignVariableOp0assignvariableop_96_layer_normalization_2_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp$assignvariableop_97_dense_3_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp"assignvariableop_98_dense_3_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_99AssignVariableOp1assignvariableop_99_layer_normalization_3_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_100AssignVariableOp1assignvariableop_100_layer_normalization_3_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp%assignvariableop_101_dense_4_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp#assignvariableop_102_dense_4_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_103AssignVariableOp2assignvariableop_103_layer_normalization_4_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_104AssignVariableOp1assignvariableop_104_layer_normalization_4_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_105AssignVariableOp%assignvariableop_105_dense_5_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_106AssignVariableOp#assignvariableop_106_dense_5_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_107AssignVariableOp2assignvariableop_107_layer_normalization_5_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_108AssignVariableOp1assignvariableop_108_layer_normalization_5_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_109AssignVariableOp%assignvariableop_109_dense_6_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_110AssignVariableOp#assignvariableop_110_dense_6_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_111AssignVariableOp2assignvariableop_111_layer_normalization_6_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_112AssignVariableOp1assignvariableop_112_layer_normalization_6_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_113AssignVariableOp%assignvariableop_113_dense_7_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp#assignvariableop_114_dense_7_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_115AssignVariableOp2assignvariableop_115_layer_normalization_7_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_116AssignVariableOp1assignvariableop_116_layer_normalization_7_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_117AssignVariableOp(assignvariableop_117_regression_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp&assignvariableop_118_regression_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_119Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_120IdentityIdentity_119:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_120Identity_120:output:0*
_input_shapesѓ
№: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ѓ
B__inference_dense_3_layer_call_and_return_conditional_losses_83956

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_81589

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ё
@__inference_dense_layer_call_and_return_conditional_losses_81255

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
W
+__inference_concatenate_layer_call_fn_83858
inputs_0
inputs_1
identityС
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_81381`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ :Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
П
r
F__inference_concatenate_layer_call_and_return_conditional_losses_83865
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ :Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
РЪ
П
@__inference_model_layer_call_and_return_conditional_losses_83630

inputsI
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ?
1layer_normalization_mul_2_readvariableop_resource: =
/layer_normalization_add_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: A
3layer_normalization_1_mul_2_readvariableop_resource: ?
1layer_normalization_1_add_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: A
3layer_normalization_2_mul_2_readvariableop_resource: ?
1layer_normalization_2_add_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource:  5
'dense_3_biasadd_readvariableop_resource: A
3layer_normalization_3_mul_2_readvariableop_resource: ?
1layer_normalization_3_add_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: A
3layer_normalization_4_mul_2_readvariableop_resource: ?
1layer_normalization_4_add_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: A
3layer_normalization_5_mul_2_readvariableop_resource: ?
1layer_normalization_5_add_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource:  5
'dense_6_biasadd_readvariableop_resource: A
3layer_normalization_6_mul_2_readvariableop_resource: ?
1layer_normalization_6_add_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource:  5
'dense_7_biasadd_readvariableop_resource: A
3layer_normalization_7_mul_2_readvariableop_resource: ?
1layer_normalization_7_add_readvariableop_resource: ;
)regression_matmul_readvariableop_resource: 8
*regression_biasadd_readvariableop_resource:
identityЂ#batch_normalization/AssignMovingAvgЂ2batch_normalization/AssignMovingAvg/ReadVariableOpЂ%batch_normalization/AssignMovingAvg_1Ђ4batch_normalization/AssignMovingAvg_1/ReadVariableOpЂ,batch_normalization/batchnorm/ReadVariableOpЂ0batch_normalization/batchnorm/mul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂdense_7/BiasAdd/ReadVariableOpЂdense_7/MatMul/ReadVariableOpЂ&layer_normalization/add/ReadVariableOpЂ(layer_normalization/mul_2/ReadVariableOpЂ(layer_normalization_1/add/ReadVariableOpЂ*layer_normalization_1/mul_2/ReadVariableOpЂ(layer_normalization_2/add/ReadVariableOpЂ*layer_normalization_2/mul_2/ReadVariableOpЂ(layer_normalization_3/add/ReadVariableOpЂ*layer_normalization_3/mul_2/ReadVariableOpЂ(layer_normalization_4/add/ReadVariableOpЂ*layer_normalization_4/mul_2/ReadVariableOpЂ(layer_normalization_5/add/ReadVariableOpЂ*layer_normalization_5/mul_2/ReadVariableOpЂ(layer_normalization_6/add/ReadVariableOpЂ*layer_normalization_6/mul_2/ReadVariableOpЂ(layer_normalization_7/add/ReadVariableOpЂ*layer_normalization_7/mul_2/ReadVariableOpЂ!regression/BiasAdd/ReadVariableOpЂ regression/MatMul/ReadVariableOp|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ї
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:Џ
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: к
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Њ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Н
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:Д
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ќ
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ў
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0У
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:К
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:­
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:І
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0А
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ў
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ \

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
layer_normalization/ShapeShapedense/Tanh:y:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :я
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapedense/Tanh:y:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ r
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ё
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ј
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ќ
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
: *
dtype0Ё
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_1/MatMulMatMullayer_normalization/add:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_1/ShapeShapedense_1/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_1/ReshapeReshapedense_1/Tanh:y:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :З
concatenate/concatConcatV2layer_normalization/add:z:0layer_normalization_1/add:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_2/ShapeShapedense_2/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_2/mul_1Mul&layer_normalization_2/mul_1/x:output:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul:z:0layer_normalization_2/mul_1:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_2/ReshapeReshapedense_2/Tanh:y:0,layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_2/mul_2/ReadVariableOpReadVariableOp3layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_2/mul_2Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_2/addAddV2layer_normalization_2/mul_2:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_3/MatMulMatMullayer_normalization_2/add:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_3/ShapeShapedense_3/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_3/mul_1Mul&layer_normalization_3/mul_1/x:output:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul:z:0layer_normalization_3/mul_1:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_3/ReshapeReshapedense_3/Tanh:y:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_3/mul_2/ReadVariableOpReadVariableOp3layer_normalization_3_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_3/mul_2Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_3/addAddV2layer_normalization_3/mul_2:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMullayer_normalization_3/add:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_4/ShapeShapedense_4/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_4/mul_1Mul&layer_normalization_4/mul_1/x:output:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul:z:0layer_normalization_4/mul_1:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_4/ReshapeReshapedense_4/Tanh:y:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_4/mul_2/ReadVariableOpReadVariableOp3layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_4/mul_2Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_4/addAddV2layer_normalization_4/mul_2:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_5/MatMulMatMullayer_normalization_4/add:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_5/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_5/mul_1Mul&layer_normalization_5/mul_1/x:output:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul:z:0layer_normalization_5/mul_1:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_5/ReshapeReshapedense_5/Tanh:y:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_5/mul_2/ReadVariableOpReadVariableOp3layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_5/mul_2Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_5/addAddV2layer_normalization_5/mul_2:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_6/MatMulMatMullayer_normalization_5/add:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_6/ShapeShapedense_6/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_6/ReshapeReshapedense_6/Tanh:y:0,layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_7/MatMulMatMullayer_normalization_6/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_7/TanhTanhdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_7/ShapeShapedense_7/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_7/ReshapeReshapedense_7/Tanh:y:0,layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 regression/MatMul/ReadVariableOpReadVariableOp)regression_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
regression/MatMulMatMullayer_normalization_7/add:z:0(regression/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!regression/BiasAdd/ReadVariableOpReadVariableOp*regression_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
regression/BiasAddBiasAddregression/MatMul:product:0)regression/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
regression/TanhTanhregression/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentityregression/Tanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_2/ReadVariableOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_2/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_2/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_2/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp"^regression/BiasAdd/ReadVariableOp!^regression/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_2/ReadVariableOp*layer_normalization_2/mul_2/ReadVariableOp2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_2/ReadVariableOp*layer_normalization_3/mul_2/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_2/ReadVariableOp*layer_normalization_4/mul_2/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_2/ReadVariableOp*layer_normalization_5/mul_2/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp2F
!regression/BiasAdd/ReadVariableOp!regression/BiasAdd/ReadVariableOp2D
 regression/MatMul/ReadVariableOp regression/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч

*__inference_regression_layer_call_fn_84300

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_regression_layer_call_and_return_conditional_losses_81784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_84078

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
јf
П
@__inference_model_layer_call_and_return_conditional_losses_82587
input_1'
batch_normalization_82491:'
batch_normalization_82493:'
batch_normalization_82495:'
batch_normalization_82497:
dense_82500: 
dense_82502: '
layer_normalization_82505: '
layer_normalization_82507: 
dense_1_82510:  
dense_1_82512: )
layer_normalization_1_82515: )
layer_normalization_1_82517: 
dense_2_82521:@ 
dense_2_82523: )
layer_normalization_2_82526: )
layer_normalization_2_82528: 
dense_3_82531:  
dense_3_82533: )
layer_normalization_3_82536: )
layer_normalization_3_82538: 
dense_4_82541:  
dense_4_82543: )
layer_normalization_4_82546: )
layer_normalization_4_82548: 
dense_5_82551:  
dense_5_82553: )
layer_normalization_5_82556: )
layer_normalization_5_82558: 
dense_6_82561:  
dense_6_82563: )
layer_normalization_6_82566: )
layer_normalization_6_82568: 
dense_7_82571:  
dense_7_82573: )
layer_normalization_7_82576: )
layer_normalization_7_82578: "
regression_82581: 
regression_82583:
identityЂ+batch_normalization/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ+layer_normalization/StatefulPartitionedCallЂ-layer_normalization_1/StatefulPartitionedCallЂ-layer_normalization_2/StatefulPartitionedCallЂ-layer_normalization_3/StatefulPartitionedCallЂ-layer_normalization_4/StatefulPartitionedCallЂ-layer_normalization_5/StatefulPartitionedCallЂ-layer_normalization_6/StatefulPartitionedCallЂ-layer_normalization_7/StatefulPartitionedCallЂ"regression/StatefulPartitionedCallе
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_82491batch_normalization_82493batch_normalization_82495batch_normalization_82497*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81217
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_82500dense_82502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_81255М
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0layer_normalization_82505layer_normalization_82507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_81303
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_82510dense_1_82512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_81320Ц
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0layer_normalization_1_82515layer_normalization_1_82517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368Ї
concatenate/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_81381
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_82521dense_2_82523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_81394Ц
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0layer_normalization_2_82526layer_normalization_2_82528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_3_82531dense_3_82533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_81459Ц
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0layer_normalization_3_82536layer_normalization_3_82538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_81507
dense_4/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_4_82541dense_4_82543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_81524Ц
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0layer_normalization_4_82546layer_normalization_4_82548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_81572
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_5_82551dense_5_82553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_81589Ц
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0layer_normalization_5_82556layer_normalization_5_82558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637
dense_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_6_82561dense_6_82563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_81654Ц
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0layer_normalization_6_82566layer_normalization_6_82568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0dense_7_82571dense_7_82573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_81719Ц
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0layer_normalization_7_82576layer_normalization_7_82578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767Ј
"regression/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0regression_82581regression_82583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_regression_layer_call_and_return_conditional_losses_81784z
IdentityIdentity+regression/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall#^regression/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2H
"regression/StatefulPartitionedCall"regression/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1


ѓ
B__inference_dense_6_layer_call_and_return_conditional_losses_84169

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
С

'__inference_dense_4_layer_call_fn_84016

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_81524o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ѓБ
н
@__inference_model_layer_call_and_return_conditional_losses_83227

inputsC
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ?
1layer_normalization_mul_2_readvariableop_resource: =
/layer_normalization_add_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: A
3layer_normalization_1_mul_2_readvariableop_resource: ?
1layer_normalization_1_add_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: A
3layer_normalization_2_mul_2_readvariableop_resource: ?
1layer_normalization_2_add_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource:  5
'dense_3_biasadd_readvariableop_resource: A
3layer_normalization_3_mul_2_readvariableop_resource: ?
1layer_normalization_3_add_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: A
3layer_normalization_4_mul_2_readvariableop_resource: ?
1layer_normalization_4_add_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: A
3layer_normalization_5_mul_2_readvariableop_resource: ?
1layer_normalization_5_add_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource:  5
'dense_6_biasadd_readvariableop_resource: A
3layer_normalization_6_mul_2_readvariableop_resource: ?
1layer_normalization_6_add_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource:  5
'dense_7_biasadd_readvariableop_resource: A
3layer_normalization_7_mul_2_readvariableop_resource: ?
1layer_normalization_7_add_readvariableop_resource: ;
)regression_matmul_readvariableop_resource: 8
*regression_biasadd_readvariableop_resource:
identityЂ,batch_normalization/batchnorm/ReadVariableOpЂ.batch_normalization/batchnorm/ReadVariableOp_1Ђ.batch_normalization/batchnorm/ReadVariableOp_2Ђ0batch_normalization/batchnorm/mul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂdense_7/BiasAdd/ReadVariableOpЂdense_7/MatMul/ReadVariableOpЂ&layer_normalization/add/ReadVariableOpЂ(layer_normalization/mul_2/ReadVariableOpЂ(layer_normalization_1/add/ReadVariableOpЂ*layer_normalization_1/mul_2/ReadVariableOpЂ(layer_normalization_2/add/ReadVariableOpЂ*layer_normalization_2/mul_2/ReadVariableOpЂ(layer_normalization_3/add/ReadVariableOpЂ*layer_normalization_3/mul_2/ReadVariableOpЂ(layer_normalization_4/add/ReadVariableOpЂ*layer_normalization_4/mul_2/ReadVariableOpЂ(layer_normalization_5/add/ReadVariableOpЂ*layer_normalization_5/mul_2/ReadVariableOpЂ(layer_normalization_6/add/ReadVariableOpЂ*layer_normalization_6/mul_2/ReadVariableOpЂ(layer_normalization_7/add/ReadVariableOpЂ*layer_normalization_7/mul_2/ReadVariableOpЂ!regression/BiasAdd/ReadVariableOpЂ regression/MatMul/ReadVariableOp
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Г
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:І
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0А
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ў
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:Ђ
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ў
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Ў
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ \

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
layer_normalization/ShapeShapedense/Tanh:y:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :я
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
layer_normalization/ReshapeReshapedense/Tanh:y:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ r
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ё
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџs
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ј
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0Ќ
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
: *
dtype0Ё
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_1/MatMulMatMullayer_normalization/add:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_1/ShapeShapedense_1/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_1/ReshapeReshapedense_1/Tanh:y:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :З
concatenate/concatConcatV2layer_normalization/add:z:0layer_normalization_1/add:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_2/ShapeShapedense_2/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_2/strided_sliceStridedSlice$layer_normalization_2/Shape:output:02layer_normalization_2/strided_slice/stack:output:04layer_normalization_2/strided_slice/stack_1:output:04layer_normalization_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_2/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_2/mulMul$layer_normalization_2/mul/x:output:0,layer_normalization_2/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_2/strided_slice_1StridedSlice$layer_normalization_2/Shape:output:04layer_normalization_2/strided_slice_1/stack:output:06layer_normalization_2/strided_slice_1/stack_1:output:06layer_normalization_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_2/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_2/mul_1Mul&layer_normalization_2/mul_1/x:output:0.layer_normalization_2/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_2/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_2/Reshape/shapePack.layer_normalization_2/Reshape/shape/0:output:0layer_normalization_2/mul:z:0layer_normalization_2/mul_1:z:0.layer_normalization_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_2/ReshapeReshapedense_2/Tanh:y:0,layer_normalization_2/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_2/ones/packedPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_2/onesFill*layer_normalization_2/ones/packed:output:0)layer_normalization_2/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_2/zeros/packedPacklayer_normalization_2/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_2/zerosFill+layer_normalization_2/zeros/packed:output:0*layer_normalization_2/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_2/FusedBatchNormV3FusedBatchNormV3&layer_normalization_2/Reshape:output:0#layer_normalization_2/ones:output:0$layer_normalization_2/zeros:output:0$layer_normalization_2/Const:output:0&layer_normalization_2/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_2/Reshape_1Reshape*layer_normalization_2/FusedBatchNormV3:y:0$layer_normalization_2/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_2/mul_2/ReadVariableOpReadVariableOp3layer_normalization_2_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_2/mul_2Mul(layer_normalization_2/Reshape_1:output:02layer_normalization_2/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_2/add/ReadVariableOpReadVariableOp1layer_normalization_2_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_2/addAddV2layer_normalization_2/mul_2:z:00layer_normalization_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_3/MatMulMatMullayer_normalization_2/add:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_3/ShapeShapedense_3/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_3/mul_1Mul&layer_normalization_3/mul_1/x:output:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul:z:0layer_normalization_3/mul_1:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_3/ReshapeReshapedense_3/Tanh:y:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_3/mul_2/ReadVariableOpReadVariableOp3layer_normalization_3_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_3/mul_2Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_3/addAddV2layer_normalization_3/mul_2:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMullayer_normalization_3/add:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_4/ShapeShapedense_4/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_4/mul_1Mul&layer_normalization_4/mul_1/x:output:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul:z:0layer_normalization_4/mul_1:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_4/ReshapeReshapedense_4/Tanh:y:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_4/mul_2/ReadVariableOpReadVariableOp3layer_normalization_4_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_4/mul_2Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_4/addAddV2layer_normalization_4/mul_2:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_5/MatMulMatMullayer_normalization_4/add:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_5/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_5/mul_1Mul&layer_normalization_5/mul_1/x:output:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul:z:0layer_normalization_5/mul_1:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_5/ReshapeReshapedense_5/Tanh:y:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_5/mul_2/ReadVariableOpReadVariableOp3layer_normalization_5_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_5/mul_2Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_5/addAddV2layer_normalization_5/mul_2:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_6/MatMulMatMullayer_normalization_5/add:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_6/ShapeShapedense_6/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_6/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_6/mul_1Mul&layer_normalization_6/mul_1/x:output:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul:z:0layer_normalization_6/mul_1:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_6/ReshapeReshapedense_6/Tanh:y:0,layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_6/mul_2/ReadVariableOpReadVariableOp3layer_normalization_6_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_6/mul_2Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_6/addAddV2layer_normalization_6/mul_2:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_7/MatMulMatMullayer_normalization_6/add:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ `
dense_7/TanhTanhdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
layer_normalization_7/ShapeShapedense_7/Tanh:y:0*
T0*
_output_shapes
:s
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_7/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :
layer_normalization_7/mul_1Mul&layer_normalization_7/mul_1/x:output:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :љ
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul:z:0layer_normalization_7/mul_1:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
layer_normalization_7/ReshapeReshapedense_7/Tanh:y:0,layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ v
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ї
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџw
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ^
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ї
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:Ў
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
*layer_normalization_7/mul_2/ReadVariableOpReadVariableOp3layer_normalization_7_mul_2_readvariableop_resource*
_output_shapes
: *
dtype0В
layer_normalization_7/mul_2Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0Ї
layer_normalization_7/addAddV2layer_normalization_7/mul_2:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
 regression/MatMul/ReadVariableOpReadVariableOp)regression_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
regression/MatMulMatMullayer_normalization_7/add:z:0(regression/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!regression/BiasAdd/ReadVariableOpReadVariableOp*regression_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
regression/BiasAddBiasAddregression/MatMul:product:0)regression/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
regression/TanhTanhregression/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentityregression/Tanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp)^layer_normalization_2/add/ReadVariableOp+^layer_normalization_2/mul_2/ReadVariableOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_2/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_2/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_2/ReadVariableOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_2/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_2/ReadVariableOp"^regression/BiasAdd/ReadVariableOp!^regression/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp2T
(layer_normalization_2/add/ReadVariableOp(layer_normalization_2/add/ReadVariableOp2X
*layer_normalization_2/mul_2/ReadVariableOp*layer_normalization_2/mul_2/ReadVariableOp2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_2/ReadVariableOp*layer_normalization_3/mul_2/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_2/ReadVariableOp*layer_normalization_4/mul_2/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_2/ReadVariableOp*layer_normalization_5/mul_2/ReadVariableOp2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_2/ReadVariableOp*layer_normalization_6/mul_2/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_2/ReadVariableOp*layer_normalization_7/mul_2/ReadVariableOp2F
!regression/BiasAdd/ReadVariableOp!regression/BiasAdd/ReadVariableOp2D
 regression/MatMul/ReadVariableOp regression/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
­
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83676

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
С

'__inference_dense_1_layer_call_fn_83790

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_81320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_1_layer_call_and_return_conditional_losses_83801

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_83852

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


і
E__inference_regression_layer_call_and_return_conditional_losses_81784

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


ѓ
B__inference_dense_7_layer_call_and_return_conditional_losses_84240

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_5_layer_call_fn_84107

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_81637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_1_layer_call_fn_83810

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_81368o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ж
в
%__inference_model_layer_call_fn_81870
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27:  

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_81791o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ч
ѓ
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_7_layer_call_fn_84249

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_81767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_84007

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
й

5__inference_layer_normalization_6_layer_call_fn_84178

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_81702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_84149

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
%
ч
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83710

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityЂAssignMovingAvgЂAssignMovingAvg/ReadVariableOpЂAssignMovingAvg_1Ђ AssignMovingAvg_1/ReadVariableOpЂbatchnorm/ReadVariableOpЂbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Ќ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Д
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

а
#__inference_signature_wrapper_82676
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27:  

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_81146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ч
ѓ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_83936

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч
ѓ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_81442

inputs+
mul_2_readvariableop_resource: )
add_readvariableop_resource: 
identityЂadd/ReadVariableOpЂmul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџK
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:џџџџџџџџџ :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
: *
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ы
­
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81170

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityЂbatchnorm/ReadVariableOpЂbatchnorm/ReadVariableOp_1Ђbatchnorm/ReadVariableOp_2Ђbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџК
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
б
%__inference_model_layer_call_fn_82757

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27:  

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_81791o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
Ю
3__inference_batch_normalization_layer_call_fn_83656

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_81217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
в
%__inference_model_layer_call_fn_82389
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19:  

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:  

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27:  

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31:  

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:
identityЂStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*F
_read_only_resource_inputs(
&$	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_82229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ>

regression0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:щА
Э
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
ъ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance"
_tf_keras_layer
Л
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
Ф
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7axis
	8gamma
9beta"
_tf_keras_layer
Л
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
Ф
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta"
_tf_keras_layer
Ѕ
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias"
_tf_keras_layer
Ф
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_axis
	`gamma
abeta"
_tf_keras_layer
Л
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
Ф
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta"
_tf_keras_layer
Л
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias"
_tf_keras_layer
Ш
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Э
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Э
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
	Ѓaxis

Єgamma
	Ѕbeta"
_tf_keras_layer
У
І	variables
Їtrainable_variables
Јregularization_losses
Љ	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses
Ќkernel
	­bias"
_tf_keras_layer
Э
Ў	variables
Џtrainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
	Дaxis

Еgamma
	Жbeta"
_tf_keras_layer
У
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Нkernel
	Оbias"
_tf_keras_layer
ж
%0
&1
'2
(3
/4
05
86
97
@8
A9
I10
J11
W12
X13
`14
a15
h16
i17
q18
r19
y20
z21
22
23
24
25
26
27
28
29
Є30
Ѕ31
Ќ32
­33
Е34
Ж35
Н36
О37"
trackable_list_wrapper
Ц
%0
&1
/2
03
84
95
@6
A7
I8
J9
W10
X11
`12
a13
h14
i15
q16
r17
y18
z19
20
21
22
23
24
25
26
27
Є28
Ѕ29
Ќ30
­31
Е32
Ж33
Н34
О35"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б
Фtrace_0
Хtrace_1
Цtrace_2
Чtrace_32о
%__inference_model_layer_call_fn_81870
%__inference_model_layer_call_fn_82757
%__inference_model_layer_call_fn_82838
%__inference_model_layer_call_fn_82389П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0zХtrace_1zЦtrace_2zЧtrace_3
Н
Шtrace_0
Щtrace_1
Ъtrace_2
Ыtrace_32Ъ
@__inference_model_layer_call_and_return_conditional_losses_83227
@__inference_model_layer_call_and_return_conditional_losses_83630
@__inference_model_layer_call_and_return_conditional_losses_82488
@__inference_model_layer_call_and_return_conditional_losses_82587П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zШtrace_0zЩtrace_1zЪtrace_2zЫtrace_3
ЫBШ
 __inference__wrapped_model_81146input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш
	Ьiter
Эbeta_1
Юbeta_2

Яdecay
аlearning_rate%mф&mх/mц0mч8mш9mщ@mъAmыImьJmэWmюXmя`m№amёhmђimѓqmєrmѕymіzmї	mј	mљ	mњ	mћ	mќ	m§	mў	mџ	Єm	Ѕm	Ќm	­m	Еm	Жm	Нm	Оm%v&v/v0v8v9v@vAvIvJvWvXv`vavhvivqvrvyvzv	v	v	v	v	v 	vЁ	vЂ	vЃ	ЄvЄ	ЅvЅ	ЌvІ	­vЇ	ЕvЈ	ЖvЉ	НvЊ	ОvЋ"
	optimizer
-
бserving_default"
signature_map
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
л
зtrace_0
иtrace_12 
3__inference_batch_normalization_layer_call_fn_83643
3__inference_batch_normalization_layer_call_fn_83656Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0zиtrace_1

йtrace_0
кtrace_12ж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83676
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83710Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0zкtrace_1
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
В
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ы
рtrace_02Ь
%__inference_dense_layer_call_fn_83719Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zрtrace_0

сtrace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_83730Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zсtrace_0
: 2dense/kernel
: 2
dense/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
В
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
љ
чtrace_02к
3__inference_layer_normalization_layer_call_fn_83739Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zчtrace_0

шtrace_02ѕ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_83781Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0
 "
trackable_list_wrapper
':% 2layer_normalization/gamma
&:$ 2layer_normalization/beta
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
э
юtrace_02Ю
'__inference_dense_1_layer_call_fn_83790Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0

яtrace_02щ
B__inference_dense_1_layer_call_and_return_conditional_losses_83801Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zяtrace_0
 :  2dense_1/kernel
: 2dense_1/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ћ
ѕtrace_02м
5__inference_layer_normalization_1_layer_call_fn_83810Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0

іtrace_02ї
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_83852Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zіtrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_1/gamma
(:& 2layer_normalization_1/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
їnon_trainable_variables
јlayers
љmetrics
 њlayer_regularization_losses
ћlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ё
ќtrace_02в
+__inference_concatenate_layer_call_fn_83858Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0

§trace_02э
F__inference_concatenate_layer_call_and_return_conditional_losses_83865Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z§trace_0
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ўnon_trainable_variables
џlayers
metrics
 layer_regularization_losses
layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_2_layer_call_fn_83874Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_2_layer_call_and_return_conditional_losses_83885Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :@ 2dense_2/kernel
: 2dense_2/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
ћ
trace_02м
5__inference_layer_normalization_2_layer_call_fn_83894Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ї
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_83936Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_2/gamma
(:& 2layer_normalization_2/beta
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_3_layer_call_fn_83945Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_3_layer_call_and_return_conditional_losses_83956Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :  2dense_3/kernel
: 2dense_3/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ћ
trace_02м
5__inference_layer_normalization_3_layer_call_fn_83965Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ї
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_84007Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_3/gamma
(:& 2layer_normalization_3/beta
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_4_layer_call_fn_84016Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

 trace_02щ
B__inference_dense_4_layer_call_and_return_conditional_losses_84027Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
 :  2dense_4/kernel
: 2dense_4/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Д
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ћ
Іtrace_02м
5__inference_layer_normalization_4_layer_call_fn_84036Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0

Їtrace_02ї
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_84078Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЇtrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_4/gamma
(:& 2layer_normalization_4/beta
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
­trace_02Ю
'__inference_dense_5_layer_call_fn_84087Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0

Ўtrace_02щ
B__inference_dense_5_layer_call_and_return_conditional_losses_84098Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0
 :  2dense_5/kernel
: 2dense_5/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ћ
Дtrace_02м
5__inference_layer_normalization_5_layer_call_fn_84107Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0

Еtrace_02ї
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_84149Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_5/gamma
(:& 2layer_normalization_5/beta
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
Лtrace_02Ю
'__inference_dense_6_layer_call_fn_84158Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0

Мtrace_02щ
B__inference_dense_6_layer_call_and_return_conditional_losses_84169Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0
 :  2dense_6/kernel
: 2dense_6/bias
0
Є0
Ѕ1"
trackable_list_wrapper
0
Є0
Ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
ћ
Тtrace_02м
5__inference_layer_normalization_6_layer_call_fn_84178Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0

Уtrace_02ї
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_84220Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_6/gamma
(:& 2layer_normalization_6/beta
0
Ќ0
­1"
trackable_list_wrapper
0
Ќ0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
І	variables
Їtrainable_variables
Јregularization_losses
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
э
Щtrace_02Ю
'__inference_dense_7_layer_call_fn_84229Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЩtrace_0

Ъtrace_02щ
B__inference_dense_7_layer_call_and_return_conditional_losses_84240Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0
 :  2dense_7/kernel
: 2dense_7/bias
0
Е0
Ж1"
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
Ў	variables
Џtrainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
ћ
аtrace_02м
5__inference_layer_normalization_7_layer_call_fn_84249Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zаtrace_0

бtrace_02ї
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_84291Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0
 "
trackable_list_wrapper
):' 2layer_normalization_7/gamma
(:& 2layer_normalization_7/beta
0
Н0
О1"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
№
зtrace_02б
*__inference_regression_layer_call_fn_84300Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0

иtrace_02ь
E__inference_regression_layer_call_and_return_conditional_losses_84311Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zиtrace_0
#:! 2regression/kernel
:2regression/bias
.
'0
(1"
trackable_list_wrapper
Ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
0
й0
к1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBє
%__inference_model_layer_call_fn_81870input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
%__inference_model_layer_call_fn_82757inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
%__inference_model_layer_call_fn_82838inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
%__inference_model_layer_call_fn_82389input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_83227inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_83630inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_82488input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_82587input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЪBЧ
#__inference_signature_wrapper_82676input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
3__inference_batch_normalization_layer_call_fn_83643inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
3__inference_batch_normalization_layer_call_fn_83656inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83676inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83710inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
%__inference_dense_layer_call_fn_83719inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
@__inference_dense_layer_call_and_return_conditional_losses_83730inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
3__inference_layer_normalization_layer_call_fn_83739inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_83781inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_1_layer_call_fn_83790inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_1_layer_call_and_return_conditional_losses_83801inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_1_layer_call_fn_83810inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_83852inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
+__inference_concatenate_layer_call_fn_83858inputs/0inputs/1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_concatenate_layer_call_and_return_conditional_losses_83865inputs/0inputs/1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_2_layer_call_fn_83874inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_2_layer_call_and_return_conditional_losses_83885inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_2_layer_call_fn_83894inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_83936inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_3_layer_call_fn_83945inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_3_layer_call_and_return_conditional_losses_83956inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_3_layer_call_fn_83965inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_84007inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_4_layer_call_fn_84016inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_84027inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_4_layer_call_fn_84036inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_84078inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_5_layer_call_fn_84087inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_5_layer_call_and_return_conditional_losses_84098inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_5_layer_call_fn_84107inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_84149inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_6_layer_call_fn_84158inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_6_layer_call_and_return_conditional_losses_84169inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_6_layer_call_fn_84178inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_84220inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_7_layer_call_fn_84229inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_7_layer_call_and_return_conditional_losses_84240inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
5__inference_layer_normalization_7_layer_call_fn_84249inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_84291inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_regression_layer_call_fn_84300inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_regression_layer_call_and_return_conditional_losses_84311inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
л	variables
м	keras_api

нtotal

оcount"
_tf_keras_metric
c
п	variables
р	keras_api

сtotal

тcount
у
_fn_kwargs"
_tf_keras_metric
0
н0
о1"
trackable_list_wrapper
.
л	variables"
_generic_user_object
:  (2total
:  (2count
0
с0
т1"
trackable_list_wrapper
.
п	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%2batch_normalization/gamma/m
&:$2batch_normalization/beta/m
: 2dense/kernel/m
: 2dense/bias/m
':% 2layer_normalization/gamma/m
&:$ 2layer_normalization/beta/m
 :  2dense_1/kernel/m
: 2dense_1/bias/m
):' 2layer_normalization_1/gamma/m
(:& 2layer_normalization_1/beta/m
 :@ 2dense_2/kernel/m
: 2dense_2/bias/m
):' 2layer_normalization_2/gamma/m
(:& 2layer_normalization_2/beta/m
 :  2dense_3/kernel/m
: 2dense_3/bias/m
):' 2layer_normalization_3/gamma/m
(:& 2layer_normalization_3/beta/m
 :  2dense_4/kernel/m
: 2dense_4/bias/m
):' 2layer_normalization_4/gamma/m
(:& 2layer_normalization_4/beta/m
 :  2dense_5/kernel/m
: 2dense_5/bias/m
):' 2layer_normalization_5/gamma/m
(:& 2layer_normalization_5/beta/m
 :  2dense_6/kernel/m
: 2dense_6/bias/m
):' 2layer_normalization_6/gamma/m
(:& 2layer_normalization_6/beta/m
 :  2dense_7/kernel/m
: 2dense_7/bias/m
):' 2layer_normalization_7/gamma/m
(:& 2layer_normalization_7/beta/m
#:! 2regression/kernel/m
:2regression/bias/m
':%2batch_normalization/gamma/v
&:$2batch_normalization/beta/v
: 2dense/kernel/v
: 2dense/bias/v
':% 2layer_normalization/gamma/v
&:$ 2layer_normalization/beta/v
 :  2dense_1/kernel/v
: 2dense_1/bias/v
):' 2layer_normalization_1/gamma/v
(:& 2layer_normalization_1/beta/v
 :@ 2dense_2/kernel/v
: 2dense_2/bias/v
):' 2layer_normalization_2/gamma/v
(:& 2layer_normalization_2/beta/v
 :  2dense_3/kernel/v
: 2dense_3/bias/v
):' 2layer_normalization_3/gamma/v
(:& 2layer_normalization_3/beta/v
 :  2dense_4/kernel/v
: 2dense_4/bias/v
):' 2layer_normalization_4/gamma/v
(:& 2layer_normalization_4/beta/v
 :  2dense_5/kernel/v
: 2dense_5/bias/v
):' 2layer_normalization_5/gamma/v
(:& 2layer_normalization_5/beta/v
 :  2dense_6/kernel/v
: 2dense_6/bias/v
):' 2layer_normalization_6/gamma/v
(:& 2layer_normalization_6/beta/v
 :  2dense_7/kernel/v
: 2dense_7/bias/v
):' 2layer_normalization_7/gamma/v
(:& 2layer_normalization_7/beta/v
#:! 2regression/kernel/v
:2regression/bias/vШ
 __inference__wrapped_model_81146Ѓ6(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "7Њ4
2

regression$!

regressionџџџџџџџџџД
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83676b(%'&3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Д
N__inference_batch_normalization_layer_call_and_return_conditional_losses_83710b'(%&3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_batch_normalization_layer_call_fn_83643U(%'&3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
3__inference_batch_normalization_layer_call_fn_83656U'(%&3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЮ
F__inference_concatenate_layer_call_and_return_conditional_losses_83865ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ@
 Ѕ
+__inference_concatenate_layer_call_fn_83858vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ 
Њ "џџџџџџџџџ@Ђ
B__inference_dense_1_layer_call_and_return_conditional_losses_83801\@A/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 z
'__inference_dense_1_layer_call_fn_83790O@A/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ђ
B__inference_dense_2_layer_call_and_return_conditional_losses_83885\WX/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 z
'__inference_dense_2_layer_call_fn_83874OWX/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Ђ
B__inference_dense_3_layer_call_and_return_conditional_losses_83956\hi/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 z
'__inference_dense_3_layer_call_fn_83945Ohi/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ђ
B__inference_dense_4_layer_call_and_return_conditional_losses_84027\yz/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 z
'__inference_dense_4_layer_call_fn_84016Oyz/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
B__inference_dense_5_layer_call_and_return_conditional_losses_84098^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
'__inference_dense_5_layer_call_fn_84087Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
B__inference_dense_6_layer_call_and_return_conditional_losses_84169^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
'__inference_dense_6_layer_call_fn_84158Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Є
B__inference_dense_7_layer_call_and_return_conditional_losses_84240^Ќ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
'__inference_dense_7_layer_call_fn_84229QЌ­/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ  
@__inference_dense_layer_call_and_return_conditional_losses_83730\/0/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ 
 x
%__inference_dense_layer_call_fn_83719O/0/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ А
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_83852\IJ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_1_layer_call_fn_83810OIJ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ А
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_83936\`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_2_layer_call_fn_83894O`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ А
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_84007\qr/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_3_layer_call_fn_83965Oqr/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ В
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_84078^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_4_layer_call_fn_84036Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ В
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_84149^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_5_layer_call_fn_84107Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ В
P__inference_layer_normalization_6_layer_call_and_return_conditional_losses_84220^ЄЅ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_6_layer_call_fn_84178QЄЅ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ В
P__inference_layer_normalization_7_layer_call_and_return_conditional_losses_84291^ЕЖ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
5__inference_layer_normalization_7_layer_call_fn_84249QЕЖ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ў
N__inference_layer_normalization_layer_call_and_return_conditional_losses_83781\89/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 
3__inference_layer_normalization_layer_call_fn_83739O89/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ о
@__inference_model_layer_call_and_return_conditional_losses_824886(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 о
@__inference_model_layer_call_and_return_conditional_losses_825876'(%&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 н
@__inference_model_layer_call_and_return_conditional_losses_832276(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 н
@__inference_model_layer_call_and_return_conditional_losses_836306'(%&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ж
%__inference_model_layer_call_fn_818706(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџЖ
%__inference_model_layer_call_fn_823896'(%&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО8Ђ5
.Ђ+
!
input_1џџџџџџџџџ
p

 
Њ "џџџџџџџџџЕ
%__inference_model_layer_call_fn_827576(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЕ
%__inference_model_layer_call_fn_828386'(%&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЇ
E__inference_regression_layer_call_and_return_conditional_losses_84311^НО/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_regression_layer_call_fn_84300QНО/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџж
#__inference_signature_wrapper_82676Ў6(%'&/089@AIJWX`ahiqryzЄЅЌ­ЕЖНО;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"7Њ4
2

regression$!

regressionџџџџџџџџџ