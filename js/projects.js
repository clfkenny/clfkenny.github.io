$(window).scroll(function() {
  if ($(document).scrollTop() > 150) {
    $('nav').addClass('shrink');
  } else {
    $('nav').removeClass('shrink');
  }
});