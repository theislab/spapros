{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}


   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if item != '__init__' %}
      ~{{ fullname }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
