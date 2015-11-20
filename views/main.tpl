<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
		"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
	<head>
		<title>Neural Network Semantic Parsing</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
    <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/js/bootstrap.min.js"></script>
    <style>
      body {
        margin: 20px 30px;
        font-size: 16px;
      }

      input {
        margin-right: 10px;
        margin-bottom: 20px;
      }
      td {
        text-align: center;
        border-style: solid;
        border-width: 1px;
        min-width: 20px;
      }
    </style>
	</head>
	<body>
    <h2>Neural Network Semantic Parsing</h2>
    <p>{{prompt}}:</p>
    <form id="query"/>
    <script type="text/javascript">
//<![CDATA[
    var form = document.getElementById("query");
    form.action="post_query"

    var input = document.createElement("input");
    input.name="query";
    input.size = 80;
    input.setAttribute("type", "text");
    form.appendChild(input);

    var button = document.createElement("input");
    button.setAttribute("type", "submit");
    // button.innerHTML = "Submit";
    button.className = "btn btn-primary";
    form.appendChild(button);
    
    var message = document.createElement("p");
    form.appendChild(message);
    button.onclick = function() {
      message.innerHTML = "Processing query...";
    }
//]]>
    </script>
    <div>
    {{!content}} 
    </div>
	</body>
</html>
