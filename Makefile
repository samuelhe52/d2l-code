.PHONY: dashboard

dashboard:
	@echo "Go to http://localhost:8000/docs/experiment_dashboard.html to view the dashboard"
	@python3 dashboard_server.py
