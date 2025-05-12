import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.shortcuts import render


def models_view(request):
    df = pd.read_csv("media/dataset_jobs.csv")

    df['Salary_avg'] = (df['Salary min'] + df['Salary max']) / 2

    le_edu = LabelEncoder()
    le_loc = LabelEncoder()
    le_job = LabelEncoder()

    df['Education Required'] = le_edu.fit_transform(df['Education Required'].astype(str))
    df['Filtered Location'] = le_loc.fit_transform(df['Filtered Location'].astype(str))
    df['Job Title'] = df['Job Title'].astype(str)

    scaler = StandardScaler()
    df[['Experience (Years)', 'Salary_avg']] = scaler.fit_transform(
        df[['Experience (Years)', 'Salary_avg']].copy()
    )

    X = df[['Education Required', 'Salary_avg', 'Experience (Years)', 'Filtered Location']]
    y = le_job.fit_transform(df['Job Title'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    context = {
        'accuracy': accuracy * 1000,
        'num_estimators': model.n_estimators,
    }

    if request.method == 'POST':
        user_input = {
            'education_required': int(request.POST.get('education_required')),
            'salary_avg': float(request.POST.get('salary_avg')),
            'experience_years': float(request.POST.get('experience_years')),
            'filtered_location': int(request.POST.get('filtered_location')),
        }

        user_scaled = scaler.transform([[user_input['experience_years'], user_input['salary_avg']]])
        user_exp_years_scaled, user_salary_scaled = user_scaled[0]

        user_vector = np.array([[
            user_input['education_required'],
            user_salary_scaled,
            user_exp_years_scaled,
            user_input['filtered_location']
        ]])

        predicted_job_encoded = model.predict(user_vector)[0]
        predicted_job = le_job.inverse_transform([predicted_job_encoded])[0]

        context.update({
            'predicted_job': predicted_job,
            'user_input': user_input
        })


    return render(request, 'core/models_view.html', context)


def visualizations(request):
    df = pd.read_csv('media/dataset_jobs.csv')
    filtered_df = df.dropna(subset=['Filtered Location'])

    top_companies = df['Company'].value_counts().head(10).reset_index()
    top_companies.columns = ['Company', 'Job Count']
    fig1 = px.bar(top_companies, x='Company', y='Job Count', title='Top Hiring Companies')

    fig2 = px.pie(df, names='Experience Required', title='Job Postings by Experience Level')

    highest_paying_jobs = df.nlargest(10, 'Salary max')[['Job Title', 'Salary max']]
    fig3 = px.bar(highest_paying_jobs, x='Job Title', y='Salary max', title='Top 10 Highest Paying Jobs')

    fig4 = px.pie(df, names='Education Required', title='Education Requirements Distribution')

    most_common_jobs = df['Job Title'].value_counts().head(5).reset_index()
    most_common_jobs.columns = ['Job Title', 'Count']
    fig5 = px.bar(most_common_jobs, x='Job Title', y='Count', title='Top 5 Most Common Job Titles')

    salary_by_category = df.groupby('Prof Area')[['Salary min', 'Salary max']].mean().reset_index()
    salary_by_category['Average Salary'] = salary_by_category[['Salary min', 'Salary max']].mean(axis=1)
    fig6 = px.bar(salary_by_category, x='Prof Area', y='Average Salary', title='Average Salary by Job Category')

    avg_salary_by_location = df.groupby("Filtered Location")[["Salary min", "Salary max"]].mean().dropna().reset_index()
    fig7 = go.Figure()
    fig7.add_trace(
        go.Bar(x=avg_salary_by_location["Filtered Location"], y=avg_salary_by_location["Salary min"], name="Min Salary",
               marker_color='orange'))
    fig7.add_trace(
        go.Bar(x=avg_salary_by_location["Filtered Location"], y=avg_salary_by_location["Salary max"], name="Max Salary",
               marker_color='green'))
    fig7.update_layout(title="Avg Min/Max Salary by Region", barmode='group', xaxis=dict(tickangle=-45),
                       template="plotly_white")

    fig8 = px.scatter(df, x="Salary min", y="Salary max", title="Min vs Max Salary Correlation",
                      color="Filtered Location", template="plotly_white")

    experience_counts = df["Experience Required"].value_counts().dropna()
    fig9 = px.bar(experience_counts, x=experience_counts.index, y=experience_counts.values,
                  labels={"x": "Experience", "y": "Count"}, title="Jobs by Experience Level",
                  color=experience_counts.values, color_continuous_scale='Purples')
    fig9.update_layout(xaxis=dict(tickangle=-45), template="plotly_white")

    location_counts = filtered_df["Filtered Location"].value_counts().dropna()
    fig10 = px.bar(location_counts, x=location_counts.index, y=location_counts.values,
                   labels={"x": "Location", "y": "Job Count"}, title="Jobs by Location",
                   color=location_counts.values, color_continuous_scale='Viridis')
    fig10.update_layout(xaxis=dict(tickangle=-45), template="plotly_white")

    prof_area_distribution = df.groupby(["Filtered Location", "Prof Area"]).size().reset_index(name="counts")
    top_10_prof_areas = prof_area_distribution["Prof Area"].value_counts().head(10).index
    filtered_prof_distribution = prof_area_distribution[prof_area_distribution["Prof Area"].isin(top_10_prof_areas)]
    fig11 = px.bar(filtered_prof_distribution, x="Prof Area", y="counts", color="Filtered Location",
                   title="Top 10 Prof Areas by Region", barmode="group")
    fig11.update_layout(xaxis=dict(tickangle=-45), template="plotly_white")

    plot_htmls = [
        fig1.to_html(full_html=False),
        fig2.to_html(full_html=False),
        fig3.to_html(full_html=False),
        fig4.to_html(full_html=False),
        fig5.to_html(full_html=False),
        fig6.to_html(full_html=False),
        fig7.to_html(full_html=False),
        fig8.to_html(full_html=False),
        fig9.to_html(full_html=False),
        fig10.to_html(full_html=False),
        fig11.to_html(full_html=False),
    ]

    return render(request, 'core/visualizations.html', {'plots': plot_htmls})


def home(request):
    return render(request, 'core/home.html')


def data_view(request):
    df = pd.read_csv('media/dataset_jobs.csv')

    head = df.to_html(classes='table table-striped', index=False)
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_types': df.dtypes.to_dict()
    }
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_html = missing.to_frame(name='Missing Values').to_html(
        classes='table table-bordered') if not missing.empty else None
    describe_html = df.describe().to_html(classes='table table-hover')

    return render(request, 'core/data_view.html', {
        'head': head,
        'info': info,
        'missing_html': missing_html,
        'describe_html': describe_html
    })
