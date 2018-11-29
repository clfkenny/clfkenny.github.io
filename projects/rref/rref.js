var row = 3
var col = 3

var originalMatrixArray = [];
var originalMatrix = [];

$('.container').html( // Append initial table
	'<table> \
	</table>');

setSize();

$('#setSize').click(function(){
	// obtian desired values of rows and columns
	row = $('#numRows').val();
	col= $('#numCols').val();

	// reset the table
	$('table').html('');
	setSize();

})

// obtaining an array of the inputed values on click
$('#calculate_btn').click(function(){
	originalMatrix = [];

	getValues();
	calculateRREF();

});


function setSize(){
	// setting up table based on desired row and column numbers
	for(var i = 0; i < row; i++){
		$('table').append('<tr id = ' + i + '></tr>');
		for(var j = 0; j < col; j++){
			$('tr#'+i).append('<td> <input type="text" name = ' + i + j + '></td>');
		}
	};

}

function getValues(){
	$('td input').each(function(){
	originalMatrixArray.push($(this).val()); // obtain the inputted data
	});	

	//convert inputted data to matrix
	for(var j = 0; j< row; j++){
		var placeholder = [];
		for(var i =0; i<col; i++){

			placeholder.push(originalMatrixArray[0]);
			originalMatrixArray.shift();
		}

		console.log(placeholder);
		originalMatrix.push(placeholder);
	}

};

function calculateRREF(){
	matrix = originalMatrix;

	newTopRow = originalMatrix[0].map(function(x){return x/originalMatrix[0][0]});
	matrix[0] = newTopRow;

	for(var i = 1; i < row; i++){
		matrix[i] = matrix[0].map(function(item, index){
			var scalarMult = matrix[i][0]/matrix[0][0];
			return item - (1/scalarMult) * matrix[i][index];
		})
	}

	console.log(matrix);
};
