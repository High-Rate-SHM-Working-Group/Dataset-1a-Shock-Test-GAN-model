
å
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
À
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ìä
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0

conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconv1d_transpose/kernel

+conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose/kernel*"
_output_shapes
:@@*
dtype0

conv1d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv1d_transpose/bias
{
)conv1d_transpose/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose/bias*
_output_shapes
:@*
dtype0

conv1d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv1d_transpose_1/kernel

-conv1d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_1/kernel*"
_output_shapes
: @*
dtype0

conv1d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv1d_transpose_1/bias

+conv1d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_1/bias*
_output_shapes
: *
dtype0

conv1d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv1d_transpose_2/kernel

-conv1d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_2/kernel*"
_output_shapes
: *
dtype0

conv1d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv1d_transpose_2/bias

+conv1d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_2/bias*
_output_shapes
:*
dtype0

conv1d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_3/kernel

-conv1d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_3/kernel*"
_output_shapes
:*
dtype0

conv1d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv1d_transpose_4/kernel

-conv1d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_4/kernel*"
_output_shapes
:*
dtype0

NoOpNoOp
£+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Þ*
valueÔ*BÑ* BÊ*
¸
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
R
2trainable_variables
3	variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
R
<trainable_variables
=	variables
>regularization_losses
?	keras_api
^

@kernel
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
^

Ekernel
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
F
0
1
"2
#3
,4
-5
66
77
@8
E9
F
0
1
"2
#3
,4
-5
66
77
@8
E9
 
­
trainable_variables
	variables
Nnon_trainable_variables
Olayer_metrics

Players
Qmetrics
Rlayer_regularization_losses
regularization_losses
 
 
 
 
­
trainable_variables
	variables
Snon_trainable_variables
Tlayer_metrics

Ulayers
Vmetrics
Wlayer_regularization_losses
regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
Xnon_trainable_variables
Ylayer_metrics

Zlayers
[metrics
\layer_regularization_losses
regularization_losses
 
 
 
­
trainable_variables
	variables
]non_trainable_variables
^layer_metrics

_layers
`metrics
alayer_regularization_losses
 regularization_losses
ca
VARIABLE_VALUEconv1d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
­
$trainable_variables
%	variables
bnon_trainable_variables
clayer_metrics

dlayers
emetrics
flayer_regularization_losses
&regularization_losses
 
 
 
­
(trainable_variables
)	variables
gnon_trainable_variables
hlayer_metrics

ilayers
jmetrics
klayer_regularization_losses
*regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv1d_transpose_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
­
.trainable_variables
/	variables
lnon_trainable_variables
mlayer_metrics

nlayers
ometrics
player_regularization_losses
0regularization_losses
 
 
 
­
2trainable_variables
3	variables
qnon_trainable_variables
rlayer_metrics

slayers
tmetrics
ulayer_regularization_losses
4regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv1d_transpose_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
­
8trainable_variables
9	variables
vnon_trainable_variables
wlayer_metrics

xlayers
ymetrics
zlayer_regularization_losses
:regularization_losses
 
 
 
­
<trainable_variables
=	variables
{non_trainable_variables
|layer_metrics

}layers
~metrics
layer_regularization_losses
>regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

@0

@0
 
²
Atrainable_variables
B	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Cregularization_losses
ec
VARIABLE_VALUEconv1d_transpose_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE

E0

E0
 
²
Ftrainable_variables
G	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Hregularization_losses
 
 
 
²
Jtrainable_variables
K	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Lregularization_losses
 
 
f
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_3Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
|
serving_default_input_4Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Í
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4dense_2/kerneldense_2/biasconv1d_transpose/kernelconv1d_transpose/biasconv1d_transpose_1/kernelconv1d_transpose_1/biasconv1d_transpose_2/kernelconv1d_transpose_2/biasconv1d_transpose_3/kernelconv1d_transpose_4/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_303373
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp+conv1d_transpose/kernel/Read/ReadVariableOp)conv1d_transpose/bias/Read/ReadVariableOp-conv1d_transpose_1/kernel/Read/ReadVariableOp+conv1d_transpose_1/bias/Read/ReadVariableOp-conv1d_transpose_2/kernel/Read/ReadVariableOp+conv1d_transpose_2/bias/Read/ReadVariableOp-conv1d_transpose_3/kernel/Read/ReadVariableOp-conv1d_transpose_4/kernel/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_303988

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasconv1d_transpose/kernelconv1d_transpose/biasconv1d_transpose_1/kernelconv1d_transpose_1/biasconv1d_transpose_2/kernelconv1d_transpose_2/biasconv1d_transpose_3/kernelconv1d_transpose_4/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_304028ú
©
¦
6__inference_conditional_generator_layer_call_fn_303425
inputs_0
inputs_1
unknown:

	unknown_0:	
	unknown_1:@@
	unknown_2:@
	unknown_3: @
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_3032222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

]
A__inference_re_lu_layer_call_and_return_conditional_losses_303897

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_303027

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿

3__inference_conv1d_transpose_3_layer_call_fn_302934

inputs
unknown:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_3029262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³1
±
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_302828

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim½
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
conv1d_transpose/ExpandDimsÖ
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dimß
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_transpose/ExpandDims_1
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2µ
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2½
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv1d_transpose°
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
squeeze_dims
2
conv1d_transpose/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdd¹
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³1
±
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_302878

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim½
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
conv1d_transpose/ExpandDimsÖ
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dimß
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_transpose/ExpandDims_1
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2µ
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2½
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_transpose°
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d_transpose/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd¹
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ú
F
*__inference_reshape_2_layer_call_fn_303922

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_3030832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý.
é
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_302972

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:
identity¢,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim½
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d_transpose/ExpandDimsÖ
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dimß
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_transpose/ExpandDims_1
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2µ
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2½
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_transpose°
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d_transpose/Squeezev
TanhTanh!conv1d_transpose/Squeeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
Z
.__inference_concatenate_1_layer_call_fn_303843
inputs_0
inputs_1
identityÕ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3029962
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
á


$__inference_signature_wrapper_303373
input_3
input_4
unknown:

	unknown_0:	
	unknown_1:@@
	unknown_2:@
	unknown_3: @
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_3027362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:QM
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Ú	
÷
C__inference_dense_2_layer_call_and_return_conditional_losses_303869

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_303907

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_302996

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_303051

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º8
®
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303308
input_4
input_3"
dense_2_303276:

dense_2_303278:	-
conv1d_transpose_303282:@@%
conv1d_transpose_303284:@/
conv1d_transpose_1_303288: @'
conv1d_transpose_1_303290: /
conv1d_transpose_2_303294: '
conv1d_transpose_2_303296:/
conv1d_transpose_3_303300:/
conv1d_transpose_4_303303:
identity¢(conv1d_transpose/StatefulPartitionedCall¢*conv1d_transpose_1/StatefulPartitionedCall¢*conv1d_transpose_2/StatefulPartitionedCall¢*conv1d_transpose_3/StatefulPartitionedCall¢*conv1d_transpose_4/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallï
concatenate_1/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3029962
concatenate_1/PartitionedCall°
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_2_303276dense_2_303278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3030082!
dense_2/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_3030272
reshape_1/PartitionedCallå
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_transpose_303282conv1d_transpose_303284*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_3027782*
(conv1d_transpose/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3030392
re_lu/PartitionedCallë
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_transpose_1_303288conv1d_transpose_1_303290*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_3028282,
*conv1d_transpose_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3030512
re_lu_1/PartitionedCallí
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_transpose_2_303294conv1d_transpose_2_303296*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_3028782,
*conv1d_transpose_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3030632
re_lu_2/PartitionedCallÐ
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_transpose_3_303300*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_3029262,
*conv1d_transpose_3/StatefulPartitionedCallã
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0conv1d_transpose_4_303303*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_3029722,
*conv1d_transpose_4/StatefulPartitionedCall
reshape_2/PartitionedCallPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_3030832
reshape_2/PartitionedCallø
IdentityIdentity"reshape_2/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
Ù
¡
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303631
inputs_0
inputs_1:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:@@>
0conv1d_transpose_biasadd_readvariableop_resource:@^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource: @@
2conv1d_transpose_1_biasadd_readvariableop_resource: ^
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource: @
2conv1d_transpose_2_biasadd_readvariableop_resource:^
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:^
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:
identity¢'conv1d_transpose/BiasAdd/ReadVariableOp¢=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp¢)conv1d_transpose_1/BiasAdd/ReadVariableOp¢?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp¢)conv1d_transpose_2/BiasAdd/ReadVariableOp¢?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis¬
concatenate_1/concatConcatV2inputs_1inputs_0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate_1/concat§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp£
dense_2/MatMulMatMulconcatenate_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddj
reshape_1/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape_1/Reshape/shape/2Ò
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape£
reshape_1/ReshapeReshapedense_2/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2
reshape_1/Reshapez
conv1d_transpose/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2È
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2Ò
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y 
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/mulr
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/add/y
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/addv
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv1d_transpose/stack/2Ì
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack¦
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dimû
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape_1/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2.
,conv1d_transpose/conv1d_transpose/ExpandDims
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpª
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim£
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@20
.conv1d_transpose/conv1d_transpose/ExpandDims_1¸
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack¼
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1¼
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice¼
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackÀ
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1À
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2£
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1°
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1 
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axisø
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concatì
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2#
!conv1d_transpose/conv1d_transposeÚ
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze¿
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv1d_transpose/BiasAdd/ReadVariableOpÚ
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2
conv1d_transpose/BiasAddy

re_lu/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2

re_lu/Relu|
conv1d_transpose_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_1/Shape
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_1/strided_slice/stack
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_1
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_2Ô
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_1/strided_slice
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice_1/stack¢
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_1¢
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_2Þ
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_1/strided_slice_1v
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_1/mul/y¨
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_1/mulz
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose_1/stack/2Ö
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/mul:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_1/stackª
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimÿ
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDimsre_lu/Relu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@20
.conv1d_transpose_1/conv1d_transpose/ExpandDims
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02A
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @22
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_1/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_1/conv1d_transpose/strided_slice_1´
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_1/conv1d_transpose/concat/values_1¤
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_1/conv1d_transpose/concat/axis
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_1/conv1d_transpose/concatõ
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2%
#conv1d_transpose_1/conv1d_transposeá
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú *
squeeze_dims
2-
+conv1d_transpose_1/conv1d_transpose/SqueezeÅ
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv1d_transpose_1/BiasAdd/ReadVariableOpã
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2
conv1d_transpose_1/BiasAdd
re_lu_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2
re_lu_1/Relu~
conv1d_transpose_2/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2Ô
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack¢
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1¢
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2Þ
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y¨
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mulz
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/stack/2Ö
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stackª
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsre_lu_1/Relu:activations:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 20
.conv1d_transpose_2/conv1d_transpose/ExpandDims
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1´
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1¤
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concatõ
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transposeá
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/SqueezeÅ
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_2/BiasAdd/ReadVariableOpã
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
conv1d_transpose_2/BiasAdd
re_lu_2/ReluRelu#conv1d_transpose_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
re_lu_2/Relu~
conv1d_transpose_3/ShapeShapere_lu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2Ô
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack¢
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1¢
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2Þ
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y¨
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/stack/2Ö
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stackª
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDimsre_lu_2/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô20
.conv1d_transpose_3/conv1d_transpose/ExpandDims
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1´
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1¤
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concatõ
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transposeá
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze¤
conv1d_transpose_3/CosCos4conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
conv1d_transpose_3/Cos~
conv1d_transpose_4/ShapeShapeconv1d_transpose_3/Cos:y:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2Ô
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack¢
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1¢
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2Þ
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y¨
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2Ö
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stackª
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDimsconv1d_transpose_3/Cos:y:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè20
.conv1d_transpose_4/conv1d_transpose/ExpandDims
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1´
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1¤
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concatõ
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transposeá
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze§
conv1d_transpose_4/TanhTanh4conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2
conv1d_transpose_4/Tanhm
reshape_2/ShapeShapeconv1d_transpose_4/Tanh:y:0*
T0*
_output_shapes
:2
reshape_2/Shape
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :'2
reshape_2/Reshape/shape/1®
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape£
reshape_2/ReshapeReshapeconv1d_transpose_4/Tanh:y:0 reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2
reshape_2/Reshapeú
IdentityIdentityreshape_2/Reshape:output:0(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ù.
é
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_302926

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:
identity¢,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim½
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d_transpose/ExpandDimsÖ
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dimß
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d_transpose/ExpandDims_1
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2µ
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2½
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d_transpose°
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
conv1d_transpose/Squeezes
CosCos!conv1d_transpose/Squeeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Cos
IdentityIdentityCos:y:0-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÒÈ
§
!__inference__wrapped_model_302736
input_4
input_3P
<conditional_generator_dense_2_matmul_readvariableop_resource:
L
=conditional_generator_dense_2_biasadd_readvariableop_resource:	r
\conditional_generator_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:@@T
Fconditional_generator_conv1d_transpose_biasadd_readvariableop_resource:@t
^conditional_generator_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource: @V
Hconditional_generator_conv1d_transpose_1_biasadd_readvariableop_resource: t
^conditional_generator_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource: V
Hconditional_generator_conv1d_transpose_2_biasadd_readvariableop_resource:t
^conditional_generator_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:t
^conditional_generator_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:
identity¢=conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOp¢Sconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOp¢Uconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOp¢Uconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp¢Uconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp¢Uconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp¢4conditional_generator/dense_2/BiasAdd/ReadVariableOp¢3conditional_generator/dense_2/MatMul/ReadVariableOp¤
/conditional_generator/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :21
/conditional_generator/concatenate_1/concat/axisì
*conditional_generator/concatenate_1/concatConcatV2input_3input_48conditional_generator/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*conditional_generator/concatenate_1/concaté
3conditional_generator/dense_2/MatMul/ReadVariableOpReadVariableOp<conditional_generator_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3conditional_generator/dense_2/MatMul/ReadVariableOpû
$conditional_generator/dense_2/MatMulMatMul3conditional_generator/concatenate_1/concat:output:0;conditional_generator/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$conditional_generator/dense_2/MatMulç
4conditional_generator/dense_2/BiasAdd/ReadVariableOpReadVariableOp=conditional_generator_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4conditional_generator/dense_2/BiasAdd/ReadVariableOpú
%conditional_generator/dense_2/BiasAddBiasAdd.conditional_generator/dense_2/MatMul:product:0<conditional_generator/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%conditional_generator/dense_2/BiasAdd¬
%conditional_generator/reshape_1/ShapeShape.conditional_generator/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:2'
%conditional_generator/reshape_1/Shape´
3conditional_generator/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3conditional_generator/reshape_1/strided_slice/stack¸
5conditional_generator/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5conditional_generator/reshape_1/strided_slice/stack_1¸
5conditional_generator/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5conditional_generator/reshape_1/strided_slice/stack_2¢
-conditional_generator/reshape_1/strided_sliceStridedSlice.conditional_generator/reshape_1/Shape:output:0<conditional_generator/reshape_1/strided_slice/stack:output:0>conditional_generator/reshape_1/strided_slice/stack_1:output:0>conditional_generator/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-conditional_generator/reshape_1/strided_slice¤
/conditional_generator/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :>21
/conditional_generator/reshape_1/Reshape/shape/1¤
/conditional_generator/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@21
/conditional_generator/reshape_1/Reshape/shape/2À
-conditional_generator/reshape_1/Reshape/shapePack6conditional_generator/reshape_1/strided_slice:output:08conditional_generator/reshape_1/Reshape/shape/1:output:08conditional_generator/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2/
-conditional_generator/reshape_1/Reshape/shapeû
'conditional_generator/reshape_1/ReshapeReshape.conditional_generator/dense_2/BiasAdd:output:06conditional_generator/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2)
'conditional_generator/reshape_1/Reshape¼
,conditional_generator/conv1d_transpose/ShapeShape0conditional_generator/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2.
,conditional_generator/conv1d_transpose/ShapeÂ
:conditional_generator/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:conditional_generator/conv1d_transpose/strided_slice/stackÆ
<conditional_generator/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<conditional_generator/conv1d_transpose/strided_slice/stack_1Æ
<conditional_generator/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<conditional_generator/conv1d_transpose/strided_slice/stack_2Ì
4conditional_generator/conv1d_transpose/strided_sliceStridedSlice5conditional_generator/conv1d_transpose/Shape:output:0Cconditional_generator/conv1d_transpose/strided_slice/stack:output:0Econditional_generator/conv1d_transpose/strided_slice/stack_1:output:0Econditional_generator/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4conditional_generator/conv1d_transpose/strided_sliceÆ
<conditional_generator/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<conditional_generator/conv1d_transpose/strided_slice_1/stackÊ
>conditional_generator/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose/strided_slice_1/stack_1Ê
>conditional_generator/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose/strided_slice_1/stack_2Ö
6conditional_generator/conv1d_transpose/strided_slice_1StridedSlice5conditional_generator/conv1d_transpose/Shape:output:0Econditional_generator/conv1d_transpose/strided_slice_1/stack:output:0Gconditional_generator/conv1d_transpose/strided_slice_1/stack_1:output:0Gconditional_generator/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6conditional_generator/conv1d_transpose/strided_slice_1
,conditional_generator/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,conditional_generator/conv1d_transpose/mul/yø
*conditional_generator/conv1d_transpose/mulMul?conditional_generator/conv1d_transpose/strided_slice_1:output:05conditional_generator/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2,
*conditional_generator/conv1d_transpose/mul
,conditional_generator/conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,conditional_generator/conv1d_transpose/add/yé
*conditional_generator/conv1d_transpose/addAddV2.conditional_generator/conv1d_transpose/mul:z:05conditional_generator/conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2,
*conditional_generator/conv1d_transpose/add¢
.conditional_generator/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.conditional_generator/conv1d_transpose/stack/2º
,conditional_generator/conv1d_transpose/stackPack=conditional_generator/conv1d_transpose/strided_slice:output:0.conditional_generator/conv1d_transpose/add:z:07conditional_generator/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,conditional_generator/conv1d_transpose/stackÒ
Fconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims/dimÓ
Bconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims0conditional_generator/reshape_1/Reshape:output:0Oconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2D
Bconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDimsË
Sconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\conditional_generator_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Sconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpÖ
Hconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimû
Dconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDims[conditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1ä
Kconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stackè
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1è
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2
Econditional_generator/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice5conditional_generator/conv1d_transpose/stack:output:0Tconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0Vconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0Vconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Econditional_generator/conv1d_transpose/conv1d_transpose/strided_sliceè
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Mconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stackì
Oconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Oconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1ì
Oconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2§
Gconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice5conditional_generator/conv1d_transpose/stack:output:0Vconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Xconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Xconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1Ü
Gconditional_generator/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gconditional_generator/conv1d_transpose/conv1d_transpose/concat/values_1Ì
Cconditional_generator/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cconditional_generator/conv1d_transpose/conv1d_transpose/concat/axisü
>conditional_generator/conv1d_transpose/conv1d_transpose/concatConcatV2Nconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice:output:0Pconditional_generator/conv1d_transpose/conv1d_transpose/concat/values_1:output:0Pconditional_generator/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0Lconditional_generator/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>conditional_generator/conv1d_transpose/conv1d_transpose/concatÚ
7conditional_generator/conv1d_transpose/conv1d_transposeConv2DBackpropInputGconditional_generator/conv1d_transpose/conv1d_transpose/concat:output:0Mconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0Kconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
29
7conditional_generator/conv1d_transpose/conv1d_transpose
?conditional_generator/conv1d_transpose/conv1d_transpose/SqueezeSqueeze@conditional_generator/conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@*
squeeze_dims
2A
?conditional_generator/conv1d_transpose/conv1d_transpose/Squeeze
=conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOpFconditional_generator_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOp²
.conditional_generator/conv1d_transpose/BiasAddBiasAddHconditional_generator/conv1d_transpose/conv1d_transpose/Squeeze:output:0Econditional_generator/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@20
.conditional_generator/conv1d_transpose/BiasAdd»
 conditional_generator/re_lu/ReluRelu7conditional_generator/conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2"
 conditional_generator/re_lu/Relu¾
.conditional_generator/conv1d_transpose_1/ShapeShape.conditional_generator/re_lu/Relu:activations:0*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_1/ShapeÆ
<conditional_generator/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<conditional_generator/conv1d_transpose_1/strided_slice/stackÊ
>conditional_generator/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_1/strided_slice/stack_1Ê
>conditional_generator/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_1/strided_slice/stack_2Ø
6conditional_generator/conv1d_transpose_1/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_1/Shape:output:0Econditional_generator/conv1d_transpose_1/strided_slice/stack:output:0Gconditional_generator/conv1d_transpose_1/strided_slice/stack_1:output:0Gconditional_generator/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6conditional_generator/conv1d_transpose_1/strided_sliceÊ
>conditional_generator/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_1/strided_slice_1/stackÎ
@conditional_generator/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_1/strided_slice_1/stack_1Î
@conditional_generator/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_1/strided_slice_1/stack_2â
8conditional_generator/conv1d_transpose_1/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_1/Shape:output:0Gconditional_generator/conv1d_transpose_1/strided_slice_1/stack:output:0Iconditional_generator/conv1d_transpose_1/strided_slice_1/stack_1:output:0Iconditional_generator/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8conditional_generator/conv1d_transpose_1/strided_slice_1¢
.conditional_generator/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.conditional_generator/conv1d_transpose_1/mul/y
,conditional_generator/conv1d_transpose_1/mulMulAconditional_generator/conv1d_transpose_1/strided_slice_1:output:07conditional_generator/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2.
,conditional_generator/conv1d_transpose_1/mul¦
0conditional_generator/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 22
0conditional_generator/conv1d_transpose_1/stack/2Ä
.conditional_generator/conv1d_transpose_1/stackPack?conditional_generator/conv1d_transpose_1/strided_slice:output:00conditional_generator/conv1d_transpose_1/mul:z:09conditional_generator/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_1/stackÖ
Hconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim×
Dconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims.conditional_generator/re_lu/Relu:activations:0Qconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2F
Dconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDimsÑ
Uconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp^conditional_generator_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02W
Uconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpÚ
Jconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim
Fconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDims]conditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Sconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2H
Fconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1è
Mconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stackì
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1ì
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2«
Gconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_1/stack:output:0Vconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Xconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Xconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2I
Gconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_sliceì
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackð
Qconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
Qconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1ð
Qconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2³
Iconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_1/stack:output:0Xconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Zconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Zconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2K
Iconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1à
Iconditional_generator/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Iconditional_generator/conv1d_transpose_1/conv1d_transpose/concat/values_1Ð
Econditional_generator/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Econditional_generator/conv1d_transpose_1/conv1d_transpose/concat/axis
@conditional_generator/conv1d_transpose_1/conv1d_transpose/concatConcatV2Pconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0Rconditional_generator/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0Rconditional_generator/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0Nconditional_generator/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2B
@conditional_generator/conv1d_transpose_1/conv1d_transpose/concatã
9conditional_generator/conv1d_transpose_1/conv1d_transposeConv2DBackpropInputIconditional_generator/conv1d_transpose_1/conv1d_transpose/concat:output:0Oconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0Mconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2;
9conditional_generator/conv1d_transpose_1/conv1d_transpose£
Aconditional_generator/conv1d_transpose_1/conv1d_transpose/SqueezeSqueezeBconditional_generator/conv1d_transpose_1/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú *
squeeze_dims
2C
Aconditional_generator/conv1d_transpose_1/conv1d_transpose/Squeeze
?conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpHconditional_generator_conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOp»
0conditional_generator/conv1d_transpose_1/BiasAddBiasAddJconditional_generator/conv1d_transpose_1/conv1d_transpose/Squeeze:output:0Gconditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 22
0conditional_generator/conv1d_transpose_1/BiasAddÂ
"conditional_generator/re_lu_1/ReluRelu9conditional_generator/conv1d_transpose_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2$
"conditional_generator/re_lu_1/ReluÀ
.conditional_generator/conv1d_transpose_2/ShapeShape0conditional_generator/re_lu_1/Relu:activations:0*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_2/ShapeÆ
<conditional_generator/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<conditional_generator/conv1d_transpose_2/strided_slice/stackÊ
>conditional_generator/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_2/strided_slice/stack_1Ê
>conditional_generator/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_2/strided_slice/stack_2Ø
6conditional_generator/conv1d_transpose_2/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_2/Shape:output:0Econditional_generator/conv1d_transpose_2/strided_slice/stack:output:0Gconditional_generator/conv1d_transpose_2/strided_slice/stack_1:output:0Gconditional_generator/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6conditional_generator/conv1d_transpose_2/strided_sliceÊ
>conditional_generator/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_2/strided_slice_1/stackÎ
@conditional_generator/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_2/strided_slice_1/stack_1Î
@conditional_generator/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_2/strided_slice_1/stack_2â
8conditional_generator/conv1d_transpose_2/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_2/Shape:output:0Gconditional_generator/conv1d_transpose_2/strided_slice_1/stack:output:0Iconditional_generator/conv1d_transpose_2/strided_slice_1/stack_1:output:0Iconditional_generator/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8conditional_generator/conv1d_transpose_2/strided_slice_1¢
.conditional_generator/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.conditional_generator/conv1d_transpose_2/mul/y
,conditional_generator/conv1d_transpose_2/mulMulAconditional_generator/conv1d_transpose_2/strided_slice_1:output:07conditional_generator/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2.
,conditional_generator/conv1d_transpose_2/mul¦
0conditional_generator/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :22
0conditional_generator/conv1d_transpose_2/stack/2Ä
.conditional_generator/conv1d_transpose_2/stackPack?conditional_generator/conv1d_transpose_2/strided_slice:output:00conditional_generator/conv1d_transpose_2/mul:z:09conditional_generator/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_2/stackÖ
Hconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimÚ
Dconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims0conditional_generator/re_lu_1/Relu:activations:0Qconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2F
Dconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDimsÑ
Uconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp^conditional_generator_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02W
Uconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpÚ
Jconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim
Fconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDims]conditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Sconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2H
Fconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1è
Mconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stackì
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1ì
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2«
Gconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_2/stack:output:0Vconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Xconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Xconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2I
Gconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_sliceì
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackð
Qconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
Qconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1ð
Qconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2³
Iconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_2/stack:output:0Xconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Zconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Zconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2K
Iconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1à
Iconditional_generator/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Iconditional_generator/conv1d_transpose_2/conv1d_transpose/concat/values_1Ð
Econditional_generator/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Econditional_generator/conv1d_transpose_2/conv1d_transpose/concat/axis
@conditional_generator/conv1d_transpose_2/conv1d_transpose/concatConcatV2Pconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Rconditional_generator/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Rconditional_generator/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0Nconditional_generator/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2B
@conditional_generator/conv1d_transpose_2/conv1d_transpose/concatã
9conditional_generator/conv1d_transpose_2/conv1d_transposeConv2DBackpropInputIconditional_generator/conv1d_transpose_2/conv1d_transpose/concat:output:0Oconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0Mconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2;
9conditional_generator/conv1d_transpose_2/conv1d_transpose£
Aconditional_generator/conv1d_transpose_2/conv1d_transpose/SqueezeSqueezeBconditional_generator/conv1d_transpose_2/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
squeeze_dims
2C
Aconditional_generator/conv1d_transpose_2/conv1d_transpose/Squeeze
?conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpHconditional_generator_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOp»
0conditional_generator/conv1d_transpose_2/BiasAddBiasAddJconditional_generator/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0Gconditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô22
0conditional_generator/conv1d_transpose_2/BiasAddÂ
"conditional_generator/re_lu_2/ReluRelu9conditional_generator/conv1d_transpose_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2$
"conditional_generator/re_lu_2/ReluÀ
.conditional_generator/conv1d_transpose_3/ShapeShape0conditional_generator/re_lu_2/Relu:activations:0*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_3/ShapeÆ
<conditional_generator/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<conditional_generator/conv1d_transpose_3/strided_slice/stackÊ
>conditional_generator/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_3/strided_slice/stack_1Ê
>conditional_generator/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_3/strided_slice/stack_2Ø
6conditional_generator/conv1d_transpose_3/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_3/Shape:output:0Econditional_generator/conv1d_transpose_3/strided_slice/stack:output:0Gconditional_generator/conv1d_transpose_3/strided_slice/stack_1:output:0Gconditional_generator/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6conditional_generator/conv1d_transpose_3/strided_sliceÊ
>conditional_generator/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_3/strided_slice_1/stackÎ
@conditional_generator/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_3/strided_slice_1/stack_1Î
@conditional_generator/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_3/strided_slice_1/stack_2â
8conditional_generator/conv1d_transpose_3/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_3/Shape:output:0Gconditional_generator/conv1d_transpose_3/strided_slice_1/stack:output:0Iconditional_generator/conv1d_transpose_3/strided_slice_1/stack_1:output:0Iconditional_generator/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8conditional_generator/conv1d_transpose_3/strided_slice_1¢
.conditional_generator/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.conditional_generator/conv1d_transpose_3/mul/y
,conditional_generator/conv1d_transpose_3/mulMulAconditional_generator/conv1d_transpose_3/strided_slice_1:output:07conditional_generator/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2.
,conditional_generator/conv1d_transpose_3/mul¦
0conditional_generator/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :22
0conditional_generator/conv1d_transpose_3/stack/2Ä
.conditional_generator/conv1d_transpose_3/stackPack?conditional_generator/conv1d_transpose_3/strided_slice:output:00conditional_generator/conv1d_transpose_3/mul:z:09conditional_generator/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_3/stackÖ
Hconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimÚ
Dconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims0conditional_generator/re_lu_2/Relu:activations:0Qconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2F
Dconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDimsÑ
Uconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp^conditional_generator_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02W
Uconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpÚ
Jconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim
Fconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDims]conditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Sconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2H
Fconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1è
Mconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stackì
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1ì
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2«
Gconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_3/stack:output:0Vconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Xconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Xconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2I
Gconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_sliceì
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackð
Qconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
Qconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1ð
Qconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2³
Iconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_3/stack:output:0Xconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Zconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Zconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2K
Iconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1à
Iconditional_generator/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Iconditional_generator/conv1d_transpose_3/conv1d_transpose/concat/values_1Ð
Econditional_generator/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Econditional_generator/conv1d_transpose_3/conv1d_transpose/concat/axis
@conditional_generator/conv1d_transpose_3/conv1d_transpose/concatConcatV2Pconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Rconditional_generator/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Rconditional_generator/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0Nconditional_generator/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2B
@conditional_generator/conv1d_transpose_3/conv1d_transpose/concatã
9conditional_generator/conv1d_transpose_3/conv1d_transposeConv2DBackpropInputIconditional_generator/conv1d_transpose_3/conv1d_transpose/concat:output:0Oconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0Mconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2;
9conditional_generator/conv1d_transpose_3/conv1d_transpose£
Aconditional_generator/conv1d_transpose_3/conv1d_transpose/SqueezeSqueezeBconditional_generator/conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè*
squeeze_dims
2C
Aconditional_generator/conv1d_transpose_3/conv1d_transpose/Squeezeæ
,conditional_generator/conv1d_transpose_3/CosCosJconditional_generator/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2.
,conditional_generator/conv1d_transpose_3/CosÀ
.conditional_generator/conv1d_transpose_4/ShapeShape0conditional_generator/conv1d_transpose_3/Cos:y:0*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_4/ShapeÆ
<conditional_generator/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<conditional_generator/conv1d_transpose_4/strided_slice/stackÊ
>conditional_generator/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_4/strided_slice/stack_1Ê
>conditional_generator/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_4/strided_slice/stack_2Ø
6conditional_generator/conv1d_transpose_4/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_4/Shape:output:0Econditional_generator/conv1d_transpose_4/strided_slice/stack:output:0Gconditional_generator/conv1d_transpose_4/strided_slice/stack_1:output:0Gconditional_generator/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6conditional_generator/conv1d_transpose_4/strided_sliceÊ
>conditional_generator/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>conditional_generator/conv1d_transpose_4/strided_slice_1/stackÎ
@conditional_generator/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_4/strided_slice_1/stack_1Î
@conditional_generator/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@conditional_generator/conv1d_transpose_4/strided_slice_1/stack_2â
8conditional_generator/conv1d_transpose_4/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_4/Shape:output:0Gconditional_generator/conv1d_transpose_4/strided_slice_1/stack:output:0Iconditional_generator/conv1d_transpose_4/strided_slice_1/stack_1:output:0Iconditional_generator/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8conditional_generator/conv1d_transpose_4/strided_slice_1¢
.conditional_generator/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :20
.conditional_generator/conv1d_transpose_4/mul/y
,conditional_generator/conv1d_transpose_4/mulMulAconditional_generator/conv1d_transpose_4/strided_slice_1:output:07conditional_generator/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2.
,conditional_generator/conv1d_transpose_4/mul¦
0conditional_generator/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :22
0conditional_generator/conv1d_transpose_4/stack/2Ä
.conditional_generator/conv1d_transpose_4/stackPack?conditional_generator/conv1d_transpose_4/strided_slice:output:00conditional_generator/conv1d_transpose_4/mul:z:09conditional_generator/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:20
.conditional_generator/conv1d_transpose_4/stackÖ
Hconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2J
Hconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimÚ
Dconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims0conditional_generator/conv1d_transpose_3/Cos:y:0Qconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2F
Dconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDimsÑ
Uconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp^conditional_generator_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02W
Uconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpÚ
Jconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim
Fconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDims]conditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Sconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2H
Fconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1è
Mconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stackì
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1ì
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2«
Gconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice7conditional_generator/conv1d_transpose_4/stack:output:0Vconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Xconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Xconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2I
Gconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_sliceì
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2Q
Oconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackð
Qconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2S
Qconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1ð
Qconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2³
Iconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice7conditional_generator/conv1d_transpose_4/stack:output:0Xconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Zconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Zconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2K
Iconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1à
Iconditional_generator/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Iconditional_generator/conv1d_transpose_4/conv1d_transpose/concat/values_1Ð
Econditional_generator/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Econditional_generator/conv1d_transpose_4/conv1d_transpose/concat/axis
@conditional_generator/conv1d_transpose_4/conv1d_transpose/concatConcatV2Pconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0Rconditional_generator/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0Rconditional_generator/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0Nconditional_generator/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2B
@conditional_generator/conv1d_transpose_4/conv1d_transpose/concatã
9conditional_generator/conv1d_transpose_4/conv1d_transposeConv2DBackpropInputIconditional_generator/conv1d_transpose_4/conv1d_transpose/concat:output:0Oconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0Mconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2;
9conditional_generator/conv1d_transpose_4/conv1d_transpose£
Aconditional_generator/conv1d_transpose_4/conv1d_transpose/SqueezeSqueezeBconditional_generator/conv1d_transpose_4/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
squeeze_dims
2C
Aconditional_generator/conv1d_transpose_4/conv1d_transpose/Squeezeé
-conditional_generator/conv1d_transpose_4/TanhTanhJconditional_generator/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2/
-conditional_generator/conv1d_transpose_4/Tanh¯
%conditional_generator/reshape_2/ShapeShape1conditional_generator/conv1d_transpose_4/Tanh:y:0*
T0*
_output_shapes
:2'
%conditional_generator/reshape_2/Shape´
3conditional_generator/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3conditional_generator/reshape_2/strided_slice/stack¸
5conditional_generator/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5conditional_generator/reshape_2/strided_slice/stack_1¸
5conditional_generator/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5conditional_generator/reshape_2/strided_slice/stack_2¢
-conditional_generator/reshape_2/strided_sliceStridedSlice.conditional_generator/reshape_2/Shape:output:0<conditional_generator/reshape_2/strided_slice/stack:output:0>conditional_generator/reshape_2/strided_slice/stack_1:output:0>conditional_generator/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-conditional_generator/reshape_2/strided_slice¥
/conditional_generator/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :'21
/conditional_generator/reshape_2/Reshape/shape/1
-conditional_generator/reshape_2/Reshape/shapePack6conditional_generator/reshape_2/strided_slice:output:08conditional_generator/reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-conditional_generator/reshape_2/Reshape/shapeû
'conditional_generator/reshape_2/ReshapeReshape1conditional_generator/conv1d_transpose_4/Tanh:y:06conditional_generator/reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2)
'conditional_generator/reshape_2/Reshapeì
IdentityIdentity0conditional_generator/reshape_2/Reshape:output:0>^conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOpT^conditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOpV^conditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOpV^conditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpV^conditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpV^conditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp5^conditional_generator/dense_2/BiasAdd/ReadVariableOp4^conditional_generator/dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2~
=conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOp=conditional_generator/conv1d_transpose/BiasAdd/ReadVariableOp2ª
Sconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpSconditional_generator/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOp?conditional_generator/conv1d_transpose_1/BiasAdd/ReadVariableOp2®
Uconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpUconditional_generator/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOp?conditional_generator/conv1d_transpose_2/BiasAdd/ReadVariableOp2®
Uconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpUconditional_generator/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2®
Uconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpUconditional_generator/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2®
Uconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpUconditional_generator/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2l
4conditional_generator/dense_2/BiasAdd/ReadVariableOp4conditional_generator/dense_2/BiasAdd/ReadVariableOp2j
3conditional_generator/dense_2/MatMul/ReadVariableOp3conditional_generator/dense_2/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
£
¤
6__inference_conditional_generator_layer_call_fn_303109
input_4
input_3
unknown:

	unknown_0:	
	unknown_1:@@
	unknown_2:@
	unknown_3: @
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinput_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_3030862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
©
¦
6__inference_conditional_generator_layer_call_fn_303399
inputs_0
inputs_1
unknown:

	unknown_0:	
	unknown_1:@@
	unknown_2:@
	unknown_3: @
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_3030862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
 

(__inference_dense_2_layer_call_fn_303859

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3030082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú	
÷
C__inference_dense_2_layer_call_and_return_conditional_losses_303008

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó2
¯
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_302778

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim½
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
conv1d_transpose/ExpandDimsÖ
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dimß
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2µ
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2½
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv1d_transpose°
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
2
conv1d_transpose/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdd¹
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î
D
(__inference_re_lu_2_layer_call_fn_303912

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3030632
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¡
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303837
inputs_0
inputs_1:
&dense_2_matmul_readvariableop_resource:
6
'dense_2_biasadd_readvariableop_resource:	\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:@@>
0conv1d_transpose_biasadd_readvariableop_resource:@^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource: @@
2conv1d_transpose_1_biasadd_readvariableop_resource: ^
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource: @
2conv1d_transpose_2_biasadd_readvariableop_resource:^
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:^
Hconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource:
identity¢'conv1d_transpose/BiasAdd/ReadVariableOp¢=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp¢)conv1d_transpose_1/BiasAdd/ReadVariableOp¢?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp¢)conv1d_transpose_2/BiasAdd/ReadVariableOp¢?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp¢?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis¬
concatenate_1/concatConcatV2inputs_1inputs_0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate_1/concat§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp£
dense_2/MatMulMatMulconcatenate_1/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddj
reshape_1/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :>2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape_1/Reshape/shape/2Ò
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape£
reshape_1/ReshapeReshapedense_2/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2
reshape_1/Reshapez
conv1d_transpose/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv1d_transpose/Shape
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2È
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv1d_transpose/strided_slice
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_1
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2Ò
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose/strided_slice_1r
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/mul/y 
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/mulr
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose/add/y
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose/addv
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv1d_transpose/stack/2Ì
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/stack¦
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :22
0conv1d_transpose/conv1d_transpose/ExpandDims/dimû
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape_1/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2.
,conv1d_transpose/conv1d_transpose/ExpandDims
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02?
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpª
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dim£
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@20
.conv1d_transpose/conv1d_transpose/ExpandDims_1¸
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5conv1d_transpose/conv1d_transpose/strided_slice/stack¼
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1¼
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask21
/conv1d_transpose/conv1d_transpose/strided_slice¼
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackÀ
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1À
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2£
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask23
1conv1d_transpose/conv1d_transpose/strided_slice_1°
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:23
1conv1d_transpose/conv1d_transpose/concat/values_1 
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-conv1d_transpose/conv1d_transpose/concat/axisø
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(conv1d_transpose/conv1d_transpose/concatì
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2#
!conv1d_transpose/conv1d_transposeÚ
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@*
squeeze_dims
2+
)conv1d_transpose/conv1d_transpose/Squeeze¿
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv1d_transpose/BiasAdd/ReadVariableOpÚ
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2
conv1d_transpose/BiasAddy

re_lu/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@2

re_lu/Relu|
conv1d_transpose_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_1/Shape
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_1/strided_slice/stack
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_1
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice/stack_2Ô
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_1/strided_slice
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_1/strided_slice_1/stack¢
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_1¢
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_1/strided_slice_1/stack_2Þ
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_1/strided_slice_1v
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_1/mul/y¨
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_1/mulz
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose_1/stack/2Ö
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/mul:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_1/stackª
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimÿ
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDimsre_lu/Relu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}@20
.conv1d_transpose_1/conv1d_transpose/ExpandDims
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02A
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @22
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_1/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_1/conv1d_transpose/strided_slice_1´
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_1/conv1d_transpose/concat/values_1¤
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_1/conv1d_transpose/concat/axis
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_1/conv1d_transpose/concatõ
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2%
#conv1d_transpose_1/conv1d_transposeá
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú *
squeeze_dims
2-
+conv1d_transpose_1/conv1d_transpose/SqueezeÅ
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv1d_transpose_1/BiasAdd/ReadVariableOpã
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2
conv1d_transpose_1/BiasAdd
re_lu_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 2
re_lu_1/Relu~
conv1d_transpose_2/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_2/Shape
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_2/strided_slice/stack
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_1
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice/stack_2Ô
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_2/strided_slice
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_2/strided_slice_1/stack¢
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_1¢
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_2/strided_slice_1/stack_2Þ
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_2/strided_slice_1v
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/mul/y¨
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_2/mulz
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_2/stack/2Ö
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/mul:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_2/stackª
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsre_lu_1/Relu:activations:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿú 20
.conv1d_transpose_2/conv1d_transpose/ExpandDims
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02A
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 22
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_2/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_2/conv1d_transpose/strided_slice_1´
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_2/conv1d_transpose/concat/values_1¤
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_2/conv1d_transpose/concat/axis
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_2/conv1d_transpose/concatõ
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_2/conv1d_transposeá
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
squeeze_dims
2-
+conv1d_transpose_2/conv1d_transpose/SqueezeÅ
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv1d_transpose_2/BiasAdd/ReadVariableOpã
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
conv1d_transpose_2/BiasAdd
re_lu_2/ReluRelu#conv1d_transpose_2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
re_lu_2/Relu~
conv1d_transpose_3/ShapeShapere_lu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv1d_transpose_3/Shape
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_3/strided_slice/stack
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_1
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice/stack_2Ô
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_3/strided_slice
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_3/strided_slice_1/stack¢
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_1¢
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_3/strided_slice_1/stack_2Þ
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_3/strided_slice_1v
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/mul/y¨
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_3/mulz
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_3/stack/2Ö
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/mul:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_3/stackª
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDimsre_lu_2/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô20
.conv1d_transpose_3/conv1d_transpose/ExpandDims
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_3/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_3/conv1d_transpose/strided_slice_1´
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_3/conv1d_transpose/concat/values_1¤
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_3/conv1d_transpose/concat/axis
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_3/conv1d_transpose/concatõ
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_3/conv1d_transposeá
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè*
squeeze_dims
2-
+conv1d_transpose_3/conv1d_transpose/Squeeze¤
conv1d_transpose_3/CosCos4conv1d_transpose_3/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
conv1d_transpose_3/Cos~
conv1d_transpose_4/ShapeShapeconv1d_transpose_3/Cos:y:0*
T0*
_output_shapes
:2
conv1d_transpose_4/Shape
&conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv1d_transpose_4/strided_slice/stack
(conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_1
(conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice/stack_2Ô
 conv1d_transpose_4/strided_sliceStridedSlice!conv1d_transpose_4/Shape:output:0/conv1d_transpose_4/strided_slice/stack:output:01conv1d_transpose_4/strided_slice/stack_1:output:01conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv1d_transpose_4/strided_slice
(conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose_4/strided_slice_1/stack¢
*conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_1¢
*conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv1d_transpose_4/strided_slice_1/stack_2Þ
"conv1d_transpose_4/strided_slice_1StridedSlice!conv1d_transpose_4/Shape:output:01conv1d_transpose_4/strided_slice_1/stack:output:03conv1d_transpose_4/strided_slice_1/stack_1:output:03conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv1d_transpose_4/strided_slice_1v
conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/mul/y¨
conv1d_transpose_4/mulMul+conv1d_transpose_4/strided_slice_1:output:0!conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2
conv1d_transpose_4/mulz
conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv1d_transpose_4/stack/2Ö
conv1d_transpose_4/stackPack)conv1d_transpose_4/strided_slice:output:0conv1d_transpose_4/mul:z:0#conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose_4/stackª
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2conv1d_transpose_4/conv1d_transpose/ExpandDims/dim
.conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDimsconv1d_transpose_3/Cos:y:0;conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè20
.conv1d_transpose_4/conv1d_transpose/ExpandDims
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp®
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim«
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0conv1d_transpose_4/conv1d_transpose/ExpandDims_1¼
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7conv1d_transpose_4/conv1d_transpose/strided_slice/stackÀ
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1À
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2§
1conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_4/stack:output:0@conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask23
1conv1d_transpose_4/conv1d_transpose/strided_sliceÀ
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackÄ
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Ä
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2¯
3conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_4/stack:output:0Bconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask25
3conv1d_transpose_4/conv1d_transpose/strided_slice_1´
3conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:25
3conv1d_transpose_4/conv1d_transpose/concat/values_1¤
/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/conv1d_transpose_4/conv1d_transpose/concat/axis
*conv1d_transpose_4/conv1d_transpose/concatConcatV2:conv1d_transpose_4/conv1d_transpose/strided_slice:output:0<conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*conv1d_transpose_4/conv1d_transpose/concatõ
#conv1d_transpose_4/conv1d_transposeConv2DBackpropInput3conv1d_transpose_4/conv1d_transpose/concat:output:09conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2%
#conv1d_transpose_4/conv1d_transposeá
+conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_4/conv1d_transpose:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*
squeeze_dims
2-
+conv1d_transpose_4/conv1d_transpose/Squeeze§
conv1d_transpose_4/TanhTanh4conv1d_transpose_4/conv1d_transpose/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2
conv1d_transpose_4/Tanhm
reshape_2/ShapeShapeconv1d_transpose_4/Tanh:y:0*
T0*
_output_shapes
:2
reshape_2/Shape
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :'2
reshape_2/Reshape/shape/1®
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape£
reshape_2/ReshapeReshapeconv1d_transpose_4/Tanh:y:0 reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2
reshape_2/Reshapeú
IdentityIdentityreshape_2/Reshape:output:0(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp@^conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2
?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
£
¤
6__inference_conditional_generator_layer_call_fn_303271
input_4
input_3
unknown:

	unknown_0:	
	unknown_1:@@
	unknown_2:@
	unknown_3: @
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinput_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_3032222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
ê
B
&__inference_re_lu_layer_call_fn_303892

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3030392
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸8
®
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303086

inputs
inputs_1"
dense_2_303009:

dense_2_303011:	-
conv1d_transpose_303029:@@%
conv1d_transpose_303031:@/
conv1d_transpose_1_303041: @'
conv1d_transpose_1_303043: /
conv1d_transpose_2_303053: '
conv1d_transpose_2_303055:/
conv1d_transpose_3_303065:/
conv1d_transpose_4_303068:
identity¢(conv1d_transpose/StatefulPartitionedCall¢*conv1d_transpose_1/StatefulPartitionedCall¢*conv1d_transpose_2/StatefulPartitionedCall¢*conv1d_transpose_3/StatefulPartitionedCall¢*conv1d_transpose_4/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallï
concatenate_1/PartitionedCallPartitionedCallinputs_1inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3029962
concatenate_1/PartitionedCall°
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_2_303009dense_2_303011*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3030082!
dense_2/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_3030272
reshape_1/PartitionedCallå
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_transpose_303029conv1d_transpose_303031*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_3027782*
(conv1d_transpose/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3030392
re_lu/PartitionedCallë
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_transpose_1_303041conv1d_transpose_1_303043*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_3028282,
*conv1d_transpose_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3030512
re_lu_1/PartitionedCallí
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_transpose_2_303053conv1d_transpose_2_303055*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_3028782,
*conv1d_transpose_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3030632
re_lu_2/PartitionedCallÐ
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_transpose_3_303065*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_3029262,
*conv1d_transpose_3/StatefulPartitionedCallã
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0conv1d_transpose_4_303068*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_3029722,
*conv1d_transpose_4/StatefulPartitionedCall
reshape_2/PartitionedCallPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_3030832
reshape_2/PartitionedCallø
IdentityIdentity"reshape_2/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

]
A__inference_re_lu_layer_call_and_return_conditional_losses_303039

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸8
®
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303222

inputs
inputs_1"
dense_2_303190:

dense_2_303192:	-
conv1d_transpose_303196:@@%
conv1d_transpose_303198:@/
conv1d_transpose_1_303202: @'
conv1d_transpose_1_303204: /
conv1d_transpose_2_303208: '
conv1d_transpose_2_303210:/
conv1d_transpose_3_303214:/
conv1d_transpose_4_303217:
identity¢(conv1d_transpose/StatefulPartitionedCall¢*conv1d_transpose_1/StatefulPartitionedCall¢*conv1d_transpose_2/StatefulPartitionedCall¢*conv1d_transpose_3/StatefulPartitionedCall¢*conv1d_transpose_4/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallï
concatenate_1/PartitionedCallPartitionedCallinputs_1inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3029962
concatenate_1/PartitionedCall°
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_2_303190dense_2_303192*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3030082!
dense_2/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_3030272
reshape_1/PartitionedCallå
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_transpose_303196conv1d_transpose_303198*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_3027782*
(conv1d_transpose/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3030392
re_lu/PartitionedCallë
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_transpose_1_303202conv1d_transpose_1_303204*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_3028282,
*conv1d_transpose_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3030512
re_lu_1/PartitionedCallí
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_transpose_2_303208conv1d_transpose_2_303210*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_3028782,
*conv1d_transpose_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3030632
re_lu_2/PartitionedCallÐ
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_transpose_3_303214*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_3029262,
*conv1d_transpose_3/StatefulPartitionedCallã
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0conv1d_transpose_4_303217*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_3029722,
*conv1d_transpose_4/StatefulPartitionedCall
reshape_2/PartitionedCallPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_3030832
reshape_2/PartitionedCallø
IdentityIdentity"reshape_2/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
¢
1__inference_conv1d_transpose_layer_call_fn_302788

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_3027782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

a
E__inference_reshape_1_layer_call_and_return_conditional_losses_303887

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :>2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_303063

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_303917

inputs
identity[
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relus
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º8
®
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303345
input_4
input_3"
dense_2_303313:

dense_2_303315:	-
conv1d_transpose_303319:@@%
conv1d_transpose_303321:@/
conv1d_transpose_1_303325: @'
conv1d_transpose_1_303327: /
conv1d_transpose_2_303331: '
conv1d_transpose_2_303333:/
conv1d_transpose_3_303337:/
conv1d_transpose_4_303340:
identity¢(conv1d_transpose/StatefulPartitionedCall¢*conv1d_transpose_1/StatefulPartitionedCall¢*conv1d_transpose_2/StatefulPartitionedCall¢*conv1d_transpose_3/StatefulPartitionedCall¢*conv1d_transpose_4/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallï
concatenate_1/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3029962
concatenate_1/PartitionedCall°
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_2_303313dense_2_303315*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3030082!
dense_2/StatefulPartitionedCallý
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_3030272
reshape_1/PartitionedCallå
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_transpose_303319conv1d_transpose_303321*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_3027782*
(conv1d_transpose/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3030392
re_lu/PartitionedCallë
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_transpose_1_303325conv1d_transpose_1_303327*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_3028282,
*conv1d_transpose_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3030512
re_lu_1/PartitionedCallí
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv1d_transpose_2_303331conv1d_transpose_2_303333*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_3028782,
*conv1d_transpose_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3030632
re_lu_2/PartitionedCallÐ
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_transpose_3_303337*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_3029262,
*conv1d_transpose_3/StatefulPartitionedCallã
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0conv1d_transpose_4_303340*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_3029722,
*conv1d_transpose_4/StatefulPartitionedCall
reshape_2/PartitionedCallPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_3030832
reshape_2/PartitionedCallø
IdentityIdentity"reshape_2/PartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3
î
D
(__inference_re_lu_1_layer_call_fn_303902

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3030512
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª#

__inference__traced_save_303988
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop6
2savev2_conv1d_transpose_kernel_read_readvariableop4
0savev2_conv1d_transpose_bias_read_readvariableop8
4savev2_conv1d_transpose_1_kernel_read_readvariableop6
2savev2_conv1d_transpose_1_bias_read_readvariableop8
4savev2_conv1d_transpose_2_kernel_read_readvariableop6
2savev2_conv1d_transpose_2_bias_read_readvariableop8
4savev2_conv1d_transpose_3_kernel_read_readvariableop8
4savev2_conv1d_transpose_4_kernel_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÉ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û
valueÑBÎB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices¾
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop2savev2_conv1d_transpose_kernel_read_readvariableop0savev2_conv1d_transpose_bias_read_readvariableop4savev2_conv1d_transpose_1_kernel_read_readvariableop2savev2_conv1d_transpose_1_bias_read_readvariableop4savev2_conv1d_transpose_2_kernel_read_readvariableop2savev2_conv1d_transpose_2_bias_read_readvariableop4savev2_conv1d_transpose_3_kernel_read_readvariableop4savev2_conv1d_transpose_4_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesq
o: :
::@@:@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
: @: 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::(	$
"
_output_shapes
::(
$
"
_output_shapes
::

_output_shapes
: 
ç
¤
3__inference_conv1d_transpose_1_layer_call_fn_302838

inputs
unknown: @
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_3028282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿

3__inference_conv1d_transpose_4_layer_call_fn_302980

inputs
unknown:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_3029722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_reshape_2_layer_call_and_return_conditional_losses_303083

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :'2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_reshape_2_layer_call_and_return_conditional_losses_303934

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :'2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapep
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ'2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_303850
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ç
¤
3__inference_conv1d_transpose_2_layer_call_fn_302888

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_3028782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
F
*__inference_reshape_1_layer_call_fn_303874

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_3030272
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö/
ü
"__inference__traced_restore_304028
file_prefix3
assignvariableop_dense_2_kernel:
.
assignvariableop_1_dense_2_bias:	@
*assignvariableop_2_conv1d_transpose_kernel:@@6
(assignvariableop_3_conv1d_transpose_bias:@B
,assignvariableop_4_conv1d_transpose_1_kernel: @8
*assignvariableop_5_conv1d_transpose_1_bias: B
,assignvariableop_6_conv1d_transpose_2_kernel: 8
*assignvariableop_7_conv1d_transpose_2_bias:B
,assignvariableop_8_conv1d_transpose_3_kernel:B
,assignvariableop_9_conv1d_transpose_4_kernel:
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ï
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û
valueÑBÎB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¯
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv1d_transpose_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3­
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv1d_transpose_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv1d_transpose_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv1d_transpose_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6±
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv1d_transpose_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¯
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv1d_transpose_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8±
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv1d_transpose_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_conv1d_transpose_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ë
serving_default×
;
input_30
serving_default_input_3:0ÿÿÿÿÿÿÿÿÿ
<
input_41
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿ>
	reshape_21
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ'tensorflow/serving/predict:
ðv
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
trainable_variables
	variables
regularization_losses
	keras_api

signatures
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"Ûr
_tf_keras_network¿r{"name": "conditional_generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "conditional_generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [62, 64]}}, "name": "reshape_1", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["conv1d_transpose", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["conv1d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_2", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["conv1d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "cos", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_3", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [5]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_4", "inbound_nodes": [[["conv1d_transpose_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [5000]}}, "name": "reshape_2", "inbound_nodes": [[["conv1d_transpose_4", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_4", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["reshape_2", 0, 0]]}, "shared_object_id": 26, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [null, 128]}, {"class_name": "TensorShape", "items": [null, 4]}]}, "is_graph_network": true, "save_spec": {"class_name": "__tuple__", "items": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128]}, "float32", "input_4"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 4]}, "float32", "input_3"]}]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "conditional_generator", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [62, 64]}}, "name": "reshape_1", "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose", "inbound_nodes": [[["reshape_1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["conv1d_transpose", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["conv1d_transpose_1", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_2", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["conv1d_transpose_2", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "cos", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_3", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [5]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv1d_transpose_4", "inbound_nodes": [[["conv1d_transpose_3", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [5000]}}, "name": "reshape_2", "inbound_nodes": [[["conv1d_transpose_4", 0, 0, {}]]], "shared_object_id": 25}], "input_layers": {"class_name": "__tuple__", "items": [["input_4", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["reshape_2", 0, 0]]}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
í"ê
_tf_keras_input_layerÊ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
©
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerþ{"name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]], "shared_object_id": 2, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 128]}]}
	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"à
_tf_keras_layerÆ{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3968, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 132}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 132]}}
»
trainable_variables
	variables
 regularization_losses
!	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [62, 64]}}, "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 6}
¦

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
__call__
+&call_and_return_all_conditional_losses"ÿ	
_tf_keras_layerå	{"name": "conv1d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["reshape_1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62, 64]}}
¶
(trainable_variables
)	variables
*regularization_losses
+	keras_api
__call__
+&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv1d_transpose", 0, 0, {}]]], "shared_object_id": 10}
©

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
__call__
+&call_and_return_all_conditional_losses"

_tf_keras_layerè	{"name": "conv1d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["re_lu", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 125, 64]}}
¼
2trainable_variables
3	variables
4regularization_losses
5	keras_api
__call__
+&call_and_return_all_conditional_losses"«
_tf_keras_layer{"name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv1d_transpose_1", 0, 0, {}]]], "shared_object_id": 14}
«

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"

_tf_keras_layerê	{"name": "conv1d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["re_lu_1", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 32]}}
¼
<trainable_variables
=	variables
>regularization_losses
?	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"«
_tf_keras_layer{"name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["conv1d_transpose_2", 0, 0, {}]]], "shared_object_id": 18}


@kernel
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"

_tf_keras_layerç	{"name": "conv1d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "cos", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["re_lu_2", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500, 16]}}
©

Ekernel
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"

_tf_keras_layerò	{"name": "conv1d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1DTranspose", "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [5]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv1d_transpose_3", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 8]}}
Å
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"´
_tf_keras_layer{"name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [5000]}}, "inbound_nodes": [[["conv1d_transpose_4", 0, 0, {}]]], "shared_object_id": 25}
f
0
1
"2
#3
,4
-5
66
77
@8
E9"
trackable_list_wrapper
f
0
1
"2
#3
,4
-5
66
77
@8
E9"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
trainable_variables
	variables
Nnon_trainable_variables
Olayer_metrics

Players
Qmetrics
Rlayer_regularization_losses
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
ªserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
Snon_trainable_variables
Tlayer_metrics

Ulayers
Vmetrics
Wlayer_regularization_losses
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_2/kernel
:2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
Xnon_trainable_variables
Ylayer_metrics

Zlayers
[metrics
\layer_regularization_losses
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
]non_trainable_variables
^layer_metrics

_layers
`metrics
alayer_regularization_losses
 regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-:+@@2conv1d_transpose/kernel
#:!@2conv1d_transpose/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
$trainable_variables
%	variables
bnon_trainable_variables
clayer_metrics

dlayers
emetrics
flayer_regularization_losses
&regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
(trainable_variables
)	variables
gnon_trainable_variables
hlayer_metrics

ilayers
jmetrics
klayer_regularization_losses
*regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:- @2conv1d_transpose_1/kernel
%:# 2conv1d_transpose_1/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
.trainable_variables
/	variables
lnon_trainable_variables
mlayer_metrics

nlayers
ometrics
player_regularization_losses
0regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
2trainable_variables
3	variables
qnon_trainable_variables
rlayer_metrics

slayers
tmetrics
ulayer_regularization_losses
4regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:- 2conv1d_transpose_2/kernel
%:#2conv1d_transpose_2/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
°
8trainable_variables
9	variables
vnon_trainable_variables
wlayer_metrics

xlayers
ymetrics
zlayer_regularization_losses
:regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
<trainable_variables
=	variables
{non_trainable_variables
|layer_metrics

}layers
~metrics
layer_regularization_losses
>regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_3/kernel
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Atrainable_variables
B	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Cregularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
/:-2conv1d_transpose_4/kernel
'
E0"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ftrainable_variables
G	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Hregularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Jtrainable_variables
K	variables
non_trainable_variables
layer_metrics
layers
metrics
 layer_regularization_losses
Lregularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

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
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
!__inference__wrapped_model_302736ß
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *O¢L
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
¦2£
6__inference_conditional_generator_layer_call_fn_303109
6__inference_conditional_generator_layer_call_fn_303399
6__inference_conditional_generator_layer_call_fn_303425
6__inference_conditional_generator_layer_call_fn_303271À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303631
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303837
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303308
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303345À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_concatenate_1_layer_call_fn_303843¢
²
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
annotationsª *
 
ó2ð
I__inference_concatenate_1_layer_call_and_return_conditional_losses_303850¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_303859¢
²
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
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_303869¢
²
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
annotationsª *
 
Ô2Ñ
*__inference_reshape_1_layer_call_fn_303874¢
²
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
annotationsª *
 
ï2ì
E__inference_reshape_1_layer_call_and_return_conditional_losses_303887¢
²
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
annotationsª *
 
2
1__inference_conv1d_transpose_layer_call_fn_302788Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_302778Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
Ð2Í
&__inference_re_lu_layer_call_fn_303892¢
²
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
annotationsª *
 
ë2è
A__inference_re_lu_layer_call_and_return_conditional_losses_303897¢
²
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
annotationsª *
 
2
3__inference_conv1d_transpose_1_layer_call_fn_302838Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 2
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_302828Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
Ò2Ï
(__inference_re_lu_1_layer_call_fn_303902¢
²
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
annotationsª *
 
í2ê
C__inference_re_lu_1_layer_call_and_return_conditional_losses_303907¢
²
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
annotationsª *
 
2
3__inference_conv1d_transpose_2_layer_call_fn_302888Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 2
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_302878Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
Ò2Ï
(__inference_re_lu_2_layer_call_fn_303912¢
²
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
annotationsª *
 
í2ê
C__inference_re_lu_2_layer_call_and_return_conditional_losses_303917¢
²
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
annotationsª *
 
2
3__inference_conv1d_transpose_3_layer_call_fn_302934Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 2
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_302926Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
3__inference_conv1d_transpose_4_layer_call_fn_302980Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 2
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_302972Ê
²
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
annotationsª **¢'
%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_reshape_2_layer_call_fn_303922¢
²
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
annotationsª *
 
ï2ì
E__inference_reshape_2_layer_call_and_return_conditional_losses_303934¢
²
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
annotationsª *
 
ÒBÏ
$__inference_signature_wrapper_303373input_3input_4"
²
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
annotationsª *
 Å
!__inference__wrapped_model_302736
"#,-67@EY¢V
O¢L
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
ª "6ª3
1
	reshape_2$!
	reshape_2ÿÿÿÿÿÿÿÿÿ'Ó
I__inference_concatenate_1_layer_call_and_return_conditional_losses_303850[¢X
Q¢N
LI
"
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ª
.__inference_concatenate_1_layer_call_fn_303843x[¢X
Q¢N
LI
"
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿí
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303308
"#,-67@Ea¢^
W¢T
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ'
 í
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303345
"#,-67@Ea¢^
W¢T
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ'
 ï
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303631
"#,-67@Ec¢`
Y¢V
L¢I
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ'
 ï
Q__inference_conditional_generator_layer_call_and_return_conditional_losses_303837
"#,-67@Ec¢`
Y¢V
L¢I
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ'
 Å
6__inference_conditional_generator_layer_call_fn_303109
"#,-67@Ea¢^
W¢T
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ'Å
6__inference_conditional_generator_layer_call_fn_303271
"#,-67@Ea¢^
W¢T
J¢G
"
input_4ÿÿÿÿÿÿÿÿÿ
!
input_3ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ'Ç
6__inference_conditional_generator_layer_call_fn_303399
"#,-67@Ec¢`
Y¢V
L¢I
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ'Ç
6__inference_conditional_generator_layer_call_fn_303425
"#,-67@Ec¢`
Y¢V
L¢I
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ'È
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_302828v,-<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
  
3__inference_conv1d_transpose_1_layer_call_fn_302838i,-<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_302878v67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
3__inference_conv1d_transpose_2_layer_call_fn_302888i67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_302926u@<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
3__inference_conv1d_transpose_3_layer_call_fn_302934h@<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
N__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_302972uE<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
3__inference_conv1d_transpose_4_layer_call_fn_302980hE<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_302778v"#<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
1__inference_conv1d_transpose_layer_call_fn_302788i"#<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¥
C__inference_dense_2_layer_call_and_return_conditional_losses_303869^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_2_layer_call_fn_303859Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¹
C__inference_re_lu_1_layer_call_and_return_conditional_losses_303907r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
(__inference_re_lu_1_layer_call_fn_303902e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¹
C__inference_re_lu_2_layer_call_and_return_conditional_losses_303917r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
(__inference_re_lu_2_layer_call_fn_303912e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
A__inference_re_lu_layer_call_and_return_conditional_losses_303897r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
&__inference_re_lu_layer_call_fn_303892e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
E__inference_reshape_1_layer_call_and_return_conditional_losses_303887]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ>@
 ~
*__inference_reshape_1_layer_call_fn_303874P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ>@¯
E__inference_reshape_2_layer_call_and_return_conditional_losses_303934f<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ'
 
*__inference_reshape_2_layer_call_fn_303922Y<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ'Ù
$__inference_signature_wrapper_303373°
"#,-67@Ej¢g
¢ 
`ª]
,
input_3!
input_3ÿÿÿÿÿÿÿÿÿ
-
input_4"
input_4ÿÿÿÿÿÿÿÿÿ"6ª3
1
	reshape_2$!
	reshape_2ÿÿÿÿÿÿÿÿÿ'