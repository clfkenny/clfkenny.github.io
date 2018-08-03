var shrunk = false;
$(window).scroll(function() {
  if ($(document).scrollTop() > 150) {
    $('nav').addClass('shrink');
    shrunk = true;
  } else {
    $('nav').removeClass('shrink');
	$('nav').removeClass('hovered');
    shrunk = false;
  }
});

nav = $('nav');

function onHover(){
  if(shrunk===true){
    nav.addClass('hovered');
  }
}

function noHover(){
  nav.removeClass('hovered');
}

nav[0].addEventListener("mouseover", onHover);
nav[0].addEventListener("mouseout", noHover);