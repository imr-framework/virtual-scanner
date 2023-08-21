/*
"""
JS funtions to deal with changes on the GUI and payload regarding the user choice
of metric or imperial measuring system
----------
Author: Marina Manso
Date: 03/27/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
*/

//Hide or show the extra fields needed for imperial system and modify the labels
function metric2Imperial(){
  if($('#imp').is(':checked')){
        $("#weightlabel").html("lbs");
        $("#heightlabel").html("ft");
        $("#extraImpFields").show();
    }
  else if($('#met').is(':checked')){
    $("#weightlabel").html("kg");
    $("#heightlabel").html("cm");
    $("#extraImpFields").hide();
  }
}

/*Before submission modify the payload to always send weight and height data
on metric system*/
function imperial2metric(data2Send){
  
  if (data2Send["measuresystem"]=="metric"){
    delete data2Send.weight2;
    delete data2Send.height2;
    delete data2Send.measuresystem;
  }
  else if (data2Send["measuresystem"]=="imperial") {
    var weight_t = data2Send.weight*0.453592 + data2Send.weight2*0.0283495231;
    data2Send.weight = weight_t.toFixed(1);
    delete data2Send.weight2;

    var height_t = data2Send.height*30.48 + data2Send.height2*2.54;
    data2Send.height = height_t.toFixed();
    delete data2Send.height2;

    delete data2Send.measuresystem;
  }
}
