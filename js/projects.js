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