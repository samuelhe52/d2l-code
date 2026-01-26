.PHONY: dashboard summary

dashboard:
	@echo "Go to http://localhost:8000/docs/experiment_dashboard.html to view the dashboard"
	@python3 dashboard_server.py

summary:
	@python3 tools/model_summary.py --model=$(model)
