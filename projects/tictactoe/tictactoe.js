$('#reset_btn').click(function(){
	reset_board(board);
})

$('.mode').hide();

$('#modes_btn').click(function(){
	$('.mode').toggle(10);
});

// $('table').click(function(){
// 	$('.mode').hide();
// })

var turn = 'X' // set initial turn

var board = [
	[,,],
	[,,],
	[,,]	
]

var game_mode = 'local';
reset_board();

$('#local').click(function(){
	$('.mode').removeClass("selected");
	$('#local').addClass("selected");
	game_mode = 'local';
	reset_board();
	turn = "X";
})

$('#easy').click(function(){
	$('.mode').removeClass("selected");
	$('#easy').addClass("selected");
	game_mode = 'easy';
	reset_board();
	turn = "X";
})




// Player hovers on square
$('td').mouseenter(function(){
	selectedSquare = $(this).children('span.hover');
	if($(this).children('span.placed').text() === ""){
		selectedSquare.text(turn);
		selectedSquare.fadeIn(0);
		$(this).addClass('hover');
	}
});

$('td').mouseleave(function(){
	selectedSquare = $(this).children('span.hover');
	selectedSquare.fadeOut(0);
	$(this).removeClass('hover');

});

// Player click on a square
$('td').click(function(){
	// First check to see if position is taken
	selectedSquare = $(this).children('span.placed');
	if(selectedSquare.text() != ""){
		alert('The space is already taken!');
	}
	else{
		// Then, how on position on board
		$(this).children('span.hover').text(''); // First remove hover icon
		selectedSquare.text(turn);
		selectedSquare.fadeIn(500);

		// Update matrix
		pos = $(this).attr('id');
		updateMatrix(turn, pos);
		
		checkWin(turn, pos);

		// alternate turns
		if(game_mode ==='local'){
			if(turn==='O'){turn = 'X';}
			else{turn ='O'};
		} else if(game_mode ==='easy'){
			easy_Move();
		}
	}

});





function easy_Move(){
	turn = 'O';

	// Retrieve available positions to place
	placedPositions = $('td span.placed');
	possiblePositions = []
	for(var i =0; i<placedPositions.length; i++){
		if(placedPositions[i].innerHTML===''){
			position = placedPositions.parent()[i].id;
			possiblePositions.push(position);
		}
	}
	console.log(possiblePositions);
	// Generate random index to choose an available position
	randIndexEasy = Math.floor(Math.random()*possiblePositions.length);
	console.log(randIndexEasy);
	pos = possiblePositions[randIndexEasy];
	
	for(var i = 0; i<placedPositions.length;i++){
		if(placedPositions.parent()[i].id === pos){
			placedPositions[i].innerHTML = turn;
			$(placedPositions[i]).fadeIn(500);
		}
	}

	updateMatrix(turn, pos);
	checkWin(turn, pos);

	turn = 'X';

}




function reset_board(){
	board = [
		[,,],
		[,,],
		[,,]	
	];
	$('td span.placed').fadeOut();
	$('td div.line').html('');
	$('td').removeClass();
	$('td span.placed').text('');

}


function updateMatrix(turn, pos){
	row = pos[0];
	col = pos[1];
	board[row][col] = turn;

}

function checkWin(turn, pos){

	row = pos[0];
	col = pos[1];

	var winType = ""
	// Check columns
	var col_triplet = 0;
	for(var i = 0; i < 3; i++){
		if(board[i][col]===turn){
			col_triplet += 1;
			if (col_triplet===3){
				winType = "col";
			}
		}
	};
	// Check rows
	var row_triplet = 0;
	for(var i = 0; i < 3; i++){
		if(board[row][i]===turn){
			row_triplet += 1;
			if( row_triplet ===3){
				winType = "row";
			}
		}
	};

	// Check diagonals
	var diag_triplet_1 = 0;
	diag_i = 0;
	diag_j = 0;
	for(var i = 0; i < 3; i++){
		if(board[diag_i][diag_j] ===turn){
			diag_triplet_1 +=1;
			if(diag_triplet_1 ===3){
				winType = "diag_1";
			}
		}
		diag_i +=1;
		diag_j +=1;
	}

	diag_i = 0;
	diag_j = 2;
	var diag_triplet_2 = 0;
	for(var i = 0; i < 3; i++){
		if(board[diag_i][diag_j] ===turn){
			diag_triplet_2 +=1;
			if(diag_triplet_2 === 3){
				winType = "diag_2";
			}
		}
		diag_i +=1;
		diag_j -=1;
	}

	// Is there a winner?
	if(winType != ""){
		drawLine(row, col, winType);

		setTimeout(function(){
			alert(turn + ' wins!')}, 500);

		$('td').addClass('inactive');
	} else{
	// If no winner when full board, then game is a draw
		if($('td span.placed').text().length === 9){
			alert('Draw...')
		}
	}
}

function drawLine(row, col, winType){
	// Draw horizontal line
	if(winType === "row"){
		for(var i =0; i < 3; i++){
			$('td#' + row + i ).addClass('win');
		}
	}

	// Draw vertical line
	if(winType === "col"){
		for(var i =0; i < 3; i++){
			$('td#' + i + col).addClass('win');
		}
	}

	// Draw diagonal lines
	diag_i = 0;
	diag_j = 0;
	if(winType === 'diag_1'){
		for(var i = 0; i<3; i++){
			$('td#' + diag_i + diag_j ).addClass('win');
			diag_i += 1;
			diag_j += 1;
		}
	}

	diag_i = 0;
	diag_j = 2;
	if(winType === 'diag_2'){
		for(var i = 0; i<3; i++){
			$('td#' + diag_i + diag_j ).addClass('win');
			diag_i += 1;
			diag_j -= 1;
		}
	}
}
