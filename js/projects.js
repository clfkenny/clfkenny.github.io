var shrunk = false;
var iScrollPos = 0;

$(window).scroll(function() {

  if ($(document).scrollTop() > 150) {
    $('nav').addClass('shrink');
    shrunk = true;
  } else {
    $('nav').removeClass('shrink');
    shrunk = false;
  }

    var iCurScrollPos = $(this).scrollTop();
    if (iCurScrollPos < iScrollPos) {
        if(shrunk){
      $('nav').removeClass('shrink');
      console.log('if');
          }
    }
    iScrollPos = iCurScrollPos;



});



// var page_y = $( document ).scrollTop();
// window.location.href = window.location.href + '?page_y=' + page_y;


// //code to handle setting page offset on load
// $(function() {
//     if ( window.location.href.indexOf( 'page_y' ) != -1 ) {
//         //gets the number from end of url
//         var match = window.location.href.split('?')[1].match( /\d+$/ );
//         var page_y = match[0];

//         //sets the page offset 
//         $( 'html, body' ).scrollTop( page_y );
//     }
// });