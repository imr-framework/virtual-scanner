$(document).on("click",".mode-option",function(){

  if ($(this).val() == "Standard") {
    $("#standard-about").show();
    $("#advanced-about").hide();
  }
  else {
    $("#standard-about").hide();
    $("#advanced-about").show();
  }

});


$(document).ready(()=>{
  const canvas= document.getElementById("emi-graph");
  console.log(canvas);
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(30, 30, 15, 0, Math.PI * 2, true); // Outer circle
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(10,30);
  ctx.lineTo(50,30);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(30,10);
  ctx.lineTo(30,50);
  ctx.stroke();

})

